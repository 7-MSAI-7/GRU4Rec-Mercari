# -*- coding: utf-8 -*-
"""
이 파일은 모델의 학습, 평가, 추천 생성과 관련된 핵심 로직을 담고 있습니다.
- `train_model_with_validation`: 모델을 학습하고 검증하며, 최적의 모델을 찾습니다.
- `evaluate_model`: 학습된 모델의 성능을 정량적인 지표로 평가합니다.
- `generate_recommendations`: 최종 모델을 사용하여 실제 추천 목록을 생성합니다.
"""
import torch
from tqdm import tqdm
import time
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import logging
import numpy as np
import pandas as pd
import src.settings as config
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def train_model_with_validation(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    n_epochs,
    k_metrics,
    device,
):
    """
    주어진 데이터로 모델을 학습하고, 각 에포크(epoch)마다 검증 세트로 성능을 평가합니다.
    가장 성능이 좋았던 시점의 모델을 최종 결과로 반환하고, 학습 과정을 시각화하여 저장합니다.

    Args:
        model (torch.nn.Module): 학습시킬 모델.
        train_loader (DataLoader): 학습용 데이터 로더.
        valid_loader (DataLoader): 검증용 데이터 로더.
        criterion: 손실 함수 (예: CrossEntropyLoss).
        optimizer: 최적화 알고리즘 (예: Adam).
        scheduler: 학습률 스케줄러.
        n_epochs (int): 총 학습 에포크 수.
        k_metrics (int): Recall@k, MRR@k 계산을 위한 k값.
        device (torch.device): 학습에 사용할 장치 (CPU 또는 GPU).

    Returns:
        torch.nn.Module: 검증 세트에서 가장 좋은 성능을 보인 모델.
    """
    start_time = time.time()
    best_val_hr = 0  # 가장 좋았던 검증 세트의 Recall@k 성능을 기록
    best_model_state = None  # 가장 좋았던 모델의 파라미터를 저장
    scaler = GradScaler()  # AMP(Automatic Mixed Precision)를 위한 스케일러. 학습 속도를 높여줍니다.
    
    # Loss 및 성능 지표(Metric)를 기록하기 위한 리스트
    train_losses, val_losses = [], []
    val_recalls, val_mrrs = [], []

    # 설정 파라미터 로깅
    logger.info(f"Training Parameters:") # 현재 학습에 사용되는 주요 설정값들을 로그로 남깁니다.
    logger.info(f"  - Epochs: {n_epochs}")
    logger.info(f"  - GRU_NUM_LAYERS: {config.GRU_NUM_LAYERS}")
    logger.info(f"  - GRU_HIDDEN_DIM: {config.GRU_HIDDEN_DIM}")
    logger.info(f"  - DROPOUT_RATE: {config.DROPOUT_RATE}")
    logger.info(f"  - Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  - Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  - Weight Decay: {config.WEIGHT_DECAY}")
    logger.info(f"  - Clip Grad Norm: {config.CLIP_GRAD_NORM}")
    logger.info(f"  - Device: {device}")

    # --- 학습 루프 시작 ---
    for epoch in range(n_epochs):
        model.train()  # 모델을 학습 모드로 설정
        epoch_loss = 0
        
        # tqdm을 사용하여 학습 진행 상황을 시각적으로 보여줍니다.
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for inputs, targets in train_iterator:
            # 입력 데이터와 정답 데이터를 지정된 장치(GPU)로 이동
            inputs = tuple(i.to(device) for i in inputs)
            targets = targets.to(device)

            optimizer.zero_grad()  # 이전 배치의 그래디언트(기울기)를 초기화

            # autocast는 특정 구간에서 자동으로 자료형을 혼합(mixed precision)하여 계산 속도를 향상시킵니다.
            with autocast(device_type="cuda"):
                outputs = model(*inputs)  # 모델을 통해 예측값 계산
                loss = criterion(outputs, targets)  # 예측값과 실제 정답 간의 손실(오차) 계산

            # 스케일러를 사용하여 손실에 대한 그래디언트를 계산 (오차 역전파)
            scaler.scale(loss).backward()
             
            # 그래디언트가 너무 커져서 학습이 불안정해지는 것을 방지(exploding gradients)하기 위해
            # 그래디언트의 크기를 일정 수준 이하로 잘라냅니다(clipping).
            scaler.unscale_(optimizer) # 스케일링된 그래디언트를 원래대로 되돌림
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
            
            # 스케일러를 사용하여 옵티마이저 단계를 수행하고 스케일을 업데이트
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            train_iterator.set_postfix(loss=loss.item())  # 진행 바에 현재 손실 값 표시
        
        avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # --- 검증 단계 ---
        # 한 에포크의 학습이 끝나면, 검증 데이터로 모델의 현재 성능을 평가합니다.
        val_loss, val_recall, val_mrr, val_accuracy = evaluate_model(
            model, valid_loader, criterion, k_metrics, device
        )
        
        # 스케줄러에 검증 손실을 전달하여 학습률을 조절합니다.
        scheduler.step(val_loss)

        # 현재 학습률을 가져와 로그로 남깁니다.
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current Learning Rate: {current_lr}")

        # 에포크별 결과를 리스트에 기록
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_recalls.append(val_recall)
        val_mrrs.append(val_mrr)

        logger.info(
            f"Epoch {epoch+1}/{n_epochs} 완료 | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Valid Loss: {val_loss:.4f} | "
            f"Valid Recall@{k_metrics}: {val_recall:.4f} | "
            f"Valid MRR@{k_metrics}: {val_mrr:.4f} | "
            f"Valid Accuracy {val_accuracy:.2f}"
        )

        # 현재 모델의 검증 성능이 이전 최고 성능보다 좋으면, 모델의 상태를 저장합니다.
        if val_recall > best_val_hr:
            best_val_hr = val_recall
            best_model_state = model.state_dict()
            logger.info(f"✨ 새로운 최고 성능 달성! (Recall: {best_val_hr:.4f}).")

    end_time = time.time()
    logger.info(f"모델 학습 끝! 총 소요 시간: {end_time - start_time:.2f}초")

    # --- 학습 과정 시각화 ---
    # 학습이 끝난 후, 에포크별 손실 및 성능 지표 변화를 그래프로 그려 파일로 저장합니다.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), val_recalls, label=f'Recall@{k_metrics}')
    plt.plot(range(1, n_epochs + 1), val_mrrs, label=f'MRR@{k_metrics}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_performance_plot.png')
    logger.info("학습 성능 그래프를 'training_performance_plot.png' 파일로 저장했습니다.")
    plt.close()
    
    # 가장 성능이 좋았던 모델의 파라미터를 다시 불러와서 최종 모델로 반환합니다.
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_model(model, data_loader, criterion, k, device):
    """
    주어진 모델을 평가 데이터셋으로 평가하고, 평균 손실, Recall@k, MRR@k, 정확도를 반환합니다.
    이 함수는 벡터화 연산을 사용하여 평가 속도를 최적화했습니다.

    Args:
        model (torch.nn.Module): 평가할 모델.
        data_loader (DataLoader): 평가용 데이터 로더.
        criterion: 손실 함수.
        k (int): 상위 k개의 추천을 평가 대상으로 함.
        device (torch.device): 평가에 사용할 장치.

    Returns:
        tuple: (평균 손실, Recall@k, MRR@k, 정확도).
    """
    model.eval()  # 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)
    total_loss = 0
    all_targets_list = []  # 모든 배치의 실제 정답을 모으는 리스트
    all_preds_list = []    # 모든 배치의 상위 k개 예측을 모으는 리스트
    num_correct = 0        # 정확하게 예측한 샘플 수
    num_samples = 0        # 전체 샘플 수

    eval_iterator = tqdm(data_loader, desc="[Evaluate]", leave=False)
    with torch.no_grad():  # 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높임
        for inputs, targets in eval_iterator:
            inputs = tuple(i.to(device) for i in inputs)
            targets = targets.to(device)

            with autocast(device_type="cuda"):
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()

            # --- Recall, MRR 계산을 위한 데이터 수집 ---
            # 모델의 출력(scores)에서 가장 높은 점수를 가진 상위 k개의 인덱스를 가져옵니다.
            _, top_k_preds_indices = torch.topk(outputs, k=k, dim=1)
            
            # 현재 배치의 정답과 예측을 리스트에 추가 (나중에 한 번에 계산하기 위함)
            all_targets_list.append(targets.cpu().numpy())
            all_preds_list.append(top_k_preds_indices.cpu().numpy())

            # --- 정확도 계산 ---
            _, predictions = outputs.max(1) # 가장 점수가 높은 1개의 예측
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    if not all_targets_list:
        return avg_loss, 0.0, 0.0, 0.0
    
    # 리스트에 모아둔 모든 정답과 예측을 하나의 큰 numpy 배열로 결합
    all_targets = np.concatenate(all_targets_list)
    all_preds = np.concatenate(all_preds_list)

    # --- 성능 지표 계산 (벡터화 연산) ---
    # Recall@k: 상위 k개 예측 중에 정답이 포함되어 있는지 확인
    # all_targets[:, None]은 (N,) 배열을 (N, 1) 배열로 만듦. 브로드캐스팅을 통해 각 정답이 예측 목록에 있는지 효율적으로 비교
    hits = (all_targets[:, None] == all_preds).any(axis=1)
    recall_at_k = np.mean(hits) if len(hits) > 0 else 0.0

    # MRR@k (Mean Reciprocal Rank): 정답이 몇 번째 순위로 예측되었는지를 평가
    # 정답을 맞춘 경우, 그 순위의 역수(1/순위)를 계산하고, 모든 샘플에 대해 평균을 냅니다.
    # np.where는 정답과 예측이 일치하는 위치의 (행, 열) 인덱스를 반환합니다.
    hit_rows, hit_cols = np.where(all_targets[:, None] == all_preds)

    # 각 행(샘플)에 대한 역수 순위(reciprocal rank)를 저장할 배열
    mrrs = np.zeros(len(all_targets))
    
    # 한 샘플에 대해 여러 정답이 있을 수 있으므로, 첫 번째로 맞춘 위치만 고려하기 위해 중복된 행을 제거
    unique_hit_rows, unique_indices = np.unique(hit_rows, return_index=True)
    
    # 첫 번째 매치의 순위 (인덱스 + 1)
    ranks = hit_cols[unique_indices] + 1.0
    
    # 해당 샘플 위치에 MRR 값(1/rank)을 할당
    mrrs[unique_hit_rows] = 1.0 / ranks
    
    mrr_at_k = np.mean(mrrs) if len(mrrs) > 0 else 0.0

    accuracy = round(float(num_correct) / float(num_samples), 2)

    return avg_loss, recall_at_k, mrr_at_k, accuracy


def generate_recommendations(
    model,
    item_event_sequences,
    top_n,
    device,
    idx_to_item_id,
    event_to_idx,
    df_item_info=None,
):
    """
    주어진 아이템-이벤트 시퀀스를 바탕으로 상위 N개의 아이템을 추천합니다.

    Args:
        model: 추천을 생성할 학습된 모델.
        item_event_sequences (list): (상품명, 이벤트 종류, 카테고리...) 튜플의 리스트.
        top_n (int): 추천할 아이템의 개수.
        device: 계산에 사용할 장치.
        idx_to_item_id (dict): 인덱스를 아이템 ID로 변환하는 사전.
        event_to_idx (dict): 이벤트를 인덱스로 변환하는 사전.
        df_item_info (pd.DataFrame, optional): 추천된 아이템의 상세 정보를 가져오기 위한 데이터프레임.

    Returns:
        pd.DataFrame: 추천된 아이템 목록과 관련 정보를 담은 데이터프레임.
    """
    # 추천 생성을 위해 별도의 SentenceTransformer 모델을 로드합니다.
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        # 입력 시퀀스의 상품명들을 임베딩으로 변환합니다.
        name_embeddings = sentence_model.encode([name for name, *_ in item_event_sequences], convert_to_tensor=True).unsqueeze(0)
        # 입력 시퀀스의 이벤트들을 인덱스로 변환합니다.
        event_ids = torch.LongTensor([[event_to_idx[event_id] for _, event_id, *_ in item_event_sequences]]).to(device)

        # 모델에 입력하여 모든 아이템에 대한 점수(score)를 계산합니다.
        scores = model(name_embeddings, event_ids).squeeze(0)

        # 가장 높은 점수를 가진 상위 n개의 아이템 인덱스를 찾습니다.
        top_scores, top_indices = torch.topk(scores, top_n)
        
        # 찾은 인덱스를 실제 아이템 ID로 변환합니다.
        recommended_item_ids = [idx_to_item_id[idx.item()] for idx in top_indices.squeeze()]

    # df_item_info가 제공되면, 추천된 아이템의 상세 정보를 결합하여 반환합니다.
    if df_item_info is not None:
        try:
            recommendations_df = df_item_info.reindex(recommended_item_ids)
            recommendations_df = recommendations_df[~recommendations_df.index.duplicated(keep='first')]
            recommendations_df['item_id'] = recommended_item_ids
            recommendations_df['score'] = top_scores.squeeze().cpu().numpy()
            return recommendations_df[['item_id', 'name', 'c0_name', 'c1_name', 'c2_name', 'score']].reset_index(drop=True)
        except Exception as e:
            logger.error(f"추천 df 생성 오류: {e}")
            pass

    # df_item_info가 없거나 오류가 발생한 경우, ID와 점수만 포함된 데이터프레임을 반환합니다.
    return pd.DataFrame({
        'item_id': recommended_item_ids,
        'score': top_scores.squeeze().cpu().numpy()
    })