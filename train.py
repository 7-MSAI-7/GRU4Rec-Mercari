# -*- coding: utf-8 -*-
"""
이 파일은 추천 시스템 모델을 학습시키는 전체 과정을 담고 있습니다.
데이터를 불러오고, 전처리하며, 모델을 정의하고, 학습시킨 후, 성능을 평가하고, 최종적으로 모델을 저장합니다.
마지막으로, 학습된 모델을 사용해 실제 추천 결과를 생성하는 예시를 보여줍니다.
"""
import os
import glob
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import logging
from src.logger_config import setup_logger

# --- Local Imports ---
# 다른 파일에 정의된 함수나 설정을 불러옵니다.
import src.settings as config  # 학습에 필요한 여러 설정값(예: 파일 경로, 모델 파라미터)을 담고 있는 파일
from src.datasets.sequence_dataset import SequenceDataset  # 데이터를 모델에 입력하기 좋은 형태로 만들어주는 클래스
from src.models.gru_model import GruModel  # 우리가 사용할 추천 모델의 구조가 정의된 클래스
from src.data_processing import load_and_preprocess_data_with_split  # 데이터를 불러오고 모델 학습에 맞게 가공하는 함수
from src.training_engine import train_model_with_validation, evaluate_model, generate_recommendations  # 모델을 학습, 평가, 추천 생성하는 함수들
from src.utils import transfer_weights, collate_fn  # 학습에 도움이 되는 추가 함수들

# GPU 메모리 및 연산 최적화 설정
# 이 설정들은 GPU를 사용할 때 더 빠른 계산을 가능하게 해줍니다.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """
    메인 함수: 전체 학습 과정을 순서대로 실행합니다.
    """
    # 로그 설정: 프로그램 실행 중 발생하는 상황들을 기록하기 위한 설정
    setup_logger()
    logger = logging.getLogger(__name__)

    # 현재 사용하는 장치(CPU 또는 GPU) 정보 출력
    logger.info(f"지금 사용하는 장치는 '{config.DEVICE}' 입니다! (GPU 사용 가능하면 'cuda'로 나와요)")

    # --- 0. 데이터 로드 ---
    logger.info("Parquet 파일 로드 중...")
    # 설정 파일에 지정된 경로에서 데이터 파일을 찾습니다.
    parquet_files = glob.glob(config.DATA_PATH)
    if not parquet_files:
        # 데이터 파일이 없으면 경고 메시지를 출력하고 프로그램을 종료합니다.
        logger.warning(f"경고: '{config.DATA_PATH}' 경로에서 Parquet 파일을 찾을 수 없습니다.")
        return

    df_full = pd.concat(map(pd.read_parquet, parquet_files), ignore_index=True)
    logger.info(f"{len(df_full)}개 행 로드 완료.")


    # --- 1. 데이터 전처리 및 분할 ---
    # 불러온 데이터를 모델 학습에 사용할 수 있도록 가공하고, 학습/검증/테스트 용으로 나눕니다.
    # 이 과정에서 각 아이템과 이벤트에 고유한 번호를 붙여줍니다.
    (
        train_samples, valid_samples, test_samples,  # 학습, 검증, 테스트용 데이터
        item_id_to_idx, event_to_idx, idx_to_item_id,  # 아이템/이벤트와 고유 번호 사이의 변환 정보
        item_idx_to_embedded_name, df_item_info, class_weights  # 아이템 정보 및 클래스 가중치
    ) = load_and_preprocess_data_with_split(
        df_full,
        min_len_for_split=config.MIN_LEN_FOR_SEQ_SPLIT,  # 사용자의 행동 시퀀스를 나누기 위한 최소 길이
    )

    # --- 메모리 관리 ---
    # 임베딩(텍스트를 숫자로 변환) 생성이 완료되었으므로, 더 이상 필요 없는 모델을 메모리에서 제거하여 자원을 확보합니다.
    logger.info("임베딩 생성이 완료되어 SentenceTransformer 모델을 메모리에서 해제합니다.")

    # --- 2. 데이터 로더 준비 ---
    # 데이터를 모델에 효율적으로 공급하기 위한 '데이터 로더'를 준비합니다.
    # SequenceDataset은 데이터를 모델에 맞는 형태로 하나씩 꺼내주는 역할을 합니다.
    train_dataset = SequenceDataset(train_samples, is_train=True)
    valid_dataset = SequenceDataset(valid_samples)
    test_dataset = SequenceDataset(test_samples)

    # 현재 컴퓨터의 CPU 코어 개수를 확인합니다.
    cpu_count = os.cpu_count() or 0
    logger.info(f"CPU 코어 개수: {cpu_count}")

    # 데이터를 불러올 때 여러 CPU 코어를 사용하여 더 빠르게 처리하도록 설정합니다.
    # 하지만 Windows 환경에서는 이 기능이 문제를 일으킬 수 있어 0으로 설정합니다.
    num_workers = 0
    logger.info(f"데이터 로더 워커 개수: {num_workers}")
    
    # pin_memory와 persistent_workers는 num_workers가 0보다 클 때,
    # 데이터를 GPU로 더 빨리 전송하기 위한 설정입니다.
    pin_memory = True if num_workers > 0 else False
    persistent_workers = True if num_workers > 0 else False
    
    # prefetch_factor는 미리 데이터를 준비해두어 학습 속도를 높이는 설정입니다.
    prefetch_factor = 2 if num_workers > 0 else None
    logger.info(f"프리패치 팩터: {prefetch_factor}")

    # DataLoader는 데이터를 '배치(batch)' 단위로 묶어 모델에 전달하는 역할을 합니다.
    # 이를 통해 학습을 더 안정적이고 효율적으로 만듭니다.
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  # 학습 데이터는 순서를 섞어 사용합니다.
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False,  # 검증 데이터는 순서를 섞지 않습니다.
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,  # 테스트 데이터도 순서를 섞지 않습니다.
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # --- 3. 모델 초기화 ---
    # 추천 모델(GruModel)을 생성하고 필요한 설정값들을 전달합니다.
    model_args = {
        "device": config.DEVICE,  # 모델이 계산을 수행할 장치 (CPU 또는 GPU)
        "name_embedding_dim": config.NAME_EMBEDDING_DIM,  # 아이템 이름 정보를 얼마나 자세하게 표현할지 결정
        "event_embedding_dim": config.EVENT_EMBEDDING_DIM,  # 이벤트(예: '클릭', '구매') 정보를 얼마나 자세하게 표현할지 결정
        "gru_hidden_dim": config.GRU_HIDDEN_DIM,  # 모델의 기억 능력(복잡성)을 결정
        "dropout_rate": config.DROPOUT_RATE,  # 모델이 학습 데이터에만 너무 치우치지 않도록(과적합 방지) 일부 정보를 무작위로 무시하는 비율
        "gru_num_layers": config.GRU_NUM_LAYERS,  # 모델의 깊이를 결정 (얼마나 많은 층을 쌓을지)
        "n_events": len(event_to_idx),  # 전체 이벤트의 종류 수
        "n_items": len(item_id_to_idx),  # 전체 아이템의 종류 수
    }
    model = GruModel(**model_args)

    # 만약 이전에 학습시킨 모델 파일이 있다면, 그 상태를 불러와서 이어서 학습합니다.
    if os.path.exists(config.MODEL_SAVE_PATH):
        logger.info("저장된 모델 가중치를 불러옵니다... 💾")
        old_state_dict = torch.load(config.MODEL_SAVE_PATH)  # 저장된 모델 가중치(파라미터)를 불러옴
        new_model_state_dict = transfer_weights(old_state_dict, model)  # 이전 모델의 가중치를 새 모델 구조에 맞게 조정
        model.load_state_dict(new_model_state_dict)  # 새 모델에 가중치를 적용
        logger.info("가중치 이전 성공! ✅")

    # 모델을 지정된 장치(GPU 또는 CPU)로 보냅니다.
    model.to(config.DEVICE)

    # 모델의 총 파라미터 수 로깅
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"모델의 총 학습 가능 파라미터 수: {total_params:,}개")
    
    # 옵티마이저(Optimizer): 모델이 정답을 더 잘 맞히도록 파라미터를 수정하는 방법을 결정 (여기서는 AdamW 사용)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 스케줄러(Scheduler): 학습이 진행됨에 따라 학습률(learning rate)을 조절하여 더 정교하게 학습하도록 도와줌
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.N_EPOCHS, eta_min=1e-6)
    
    # 손실 함수(Loss Function): 모델의 예측이 실제 정답과 얼마나 다른지(오차)를 계산하는 함수
    # 계산된 클래스 가중치를 GPU로 이동하여 손실 함수에 적용합니다.
    class_weights_tensor = class_weights.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=0, label_smoothing=0.1)
    
    # --- 4. 모델 학습 및 검증 ---
    logger.info("모델 학습을 시작합니다... 🚀")
    # 설정된 값들을 바탕으로 모델 학습을 진행합니다.
    trained_model = train_model_with_validation(
        model=model,  # 우리가 만든 모델
        train_loader=train_loader,  # 학습용 데이터 로더
        valid_loader=valid_loader,  # 검증용 데이터 로더
        criterion=criterion,  # 손실 함수
        optimizer=optimizer,  # 옵티마이저
        scheduler=scheduler,  # 스케줄러
        n_epochs=config.N_EPOCHS,  # 전체 학습 데이터를 몇 번 반복해서 볼지 결정
        k_metrics=config.K_FOR_METRICS,  # 상위 몇 개의 추천 중 정답이 있는지 평가할지 결정
        device=config.DEVICE,  # 계산 장치
        accumulation_steps=config.ACCUMULATION_STEPS,
    )
    logger.info("모델 학습 완료! 🎉")

    # --- 5. 최종 모델 평가 ---
    logger.info("최종 모델 성능을 평가합니다... 📊")
    # 학습이 끝난 모델을 테스트 데이터를 이용해 최종 성능을 확인합니다.
    test_loss, test_recall, test_mrr, test_accuracy = evaluate_model(
        trained_model, test_loader, criterion, k=config.K_FOR_METRICS, device=config.DEVICE
    )
    logger.info(
        f"최종 테스트 결과: Loss: {test_loss:.4f} | "
        f"Recall@{config.K_FOR_METRICS}: {test_recall:.4f} | "  # 상위 K개 추천 중 실제 정답이 포함된 비율
        f"MRR@{config.K_FOR_METRICS}: {test_mrr:.4f} | "      # 정답 순위의 역수 평균 (정답을 얼마나 빨리 맞추는지)
        f"Accuracy: {test_accuracy:.2f}"                     # 모델이 얼마나 정확하게 예측하는지
    )

    # --- 6. 모델 저장 ---
    logger.info(f"학습된 모델을 '{config.MODEL_SAVE_PATH}'에 저장합니다... 💾")
    # 학습이 완료된 모델의 파라미터들을 파일로 저장하여 나중에 다시 사용할 수 있도록 합니다.
    torch.save(trained_model.state_dict(), config.MODEL_SAVE_PATH)
    logger.info("모델 저장 완료! ✅")

    # --- 7. 추천 생성 및 결과 확인 (예시) ---
    # 테스트 데이터가 있을 경우, 이를 이용해 실제 추천 결과를 만들어보는 예시입니다.
    if test_samples:
        logger.info("추천 결과 생성 예시... 🛍️")
        
        # 아이템 정보에서 무작위로 1~10개의 아이템-이벤트 시퀀스를 샘플링합니다.
        item_event_sequences = df_item_info.sample(n=random.randint(1, 10))
        # 샘플링된 데이터를 모델 입력 형식에 맞게 변환합니다.
        item_event_sequences = item_event_sequences.apply(
            lambda x: (
                x["name"],  # 상품명
                random.choice(list(event_to_idx.keys())[1:]),  # 랜덤 행동 ('클릭', '구매' 등)
                x["c0_name"],  # 대분류 카테고리
                x["c1_name"],  # 중분류 카테고리
                x["c2_name"],  # 소분류 카테고리
            ),
            axis=1,
        ).tolist()
        
        # 모델에 입력으로 사용할 시퀀스 정보를 출력합니다.
        logger.info(f"추천 생성을 위한 입력 시퀀스:")
        for item_event_sequence in item_event_sequences:
            logger.info(
                f"  - 상품명: {item_event_sequence[0]:>80}, 행동: {item_event_sequence[1]:>20}"
                f"  - 카테고리: {(item_event_sequence[2] if item_event_sequence[2] else ' '):<20}, "
                f"{(item_event_sequence[3] if item_event_sequence[3] else ' '):<20}, " 
                f"{(item_event_sequence[4] if item_event_sequence[4] else ' '):<20}"
            )

        # 학습된 모델을 사용하여 추천 상품 목록을 생성합니다.
        recommendations = generate_recommendations(
            model=trained_model,
            item_event_sequences=item_event_sequences,  # 입력 시퀀스
            top_n=config.TOP_N,  # 상위 몇 개를 추천할지 결정
            device=config.DEVICE,
            idx_to_item_id=idx_to_item_id,
            event_to_idx=event_to_idx,
            df_item_info=df_item_info,
        )

        # 생성된 추천 결과를 표 형태로 출력합니다.
        logger.info(f"\n--- 상위 {config.TOP_N}개 추천 상품 ---")
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            logger.info(f"\n{recommendations.to_string()}")
        else:
            logger.warning("추천 상품 정보를 불러올 수 없습니다.")

    # GPU 메모리를 정리합니다.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    logger.info("\n--- 모든 작업 완료 ---")


if __name__ == "__main__":
    # 이 스크립트가 직접 실행될 때 main() 함수를 호출합니다.
    main()
