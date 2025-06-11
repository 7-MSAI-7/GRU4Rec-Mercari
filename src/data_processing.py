# -*- coding: utf-8 -*-
"""
이 파일은 데이터 전처리와 관련된 함수들을 담고 있습니다.
주요 기능은 원본 데이터를 불러와 모델 학습에 적합한 형태로 가공하고,
학습, 검증, 테스트 데이터로 분할하는 것입니다.
"""
import os
import gc
import time
import pickle
import random
import pandas as pd
import torch
import src.settings as config
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer

# pandas에서 progress bar를 사용하기 위한 설정
tqdm.pandas()
logger = logging.getLogger(__name__)

def load_and_preprocess_data_with_split(df_full, min_len_for_split):
    """
    데이터프레임을 받아 필요한 정보만 골라내고, 모델 학습에 적합한 형태로 가공한 뒤,
    학습/검증/테스트 세트로 분할합니다.

    Args:
        df_full (pd.DataFrame): 전처리할 전체 데이터프레임.
        min_len_for_split (int): 학습/검증/테스트 분할을 위한 최소 원본 시퀀스 길이.

    Returns:
        tuple: (학습 샘플, 검증 샘플, 테스트 샘플), 각종 맵핑 사전, 아이템 정보 데이터프레임 등을 포함하는 튜플.
    """
    # SentenceTransformer 모델 로드: 텍스트(상품명)를 숫자 벡터(임베딩)로 변환합니다.
    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    start_time = time.time()
    logger.info("데이터 전처리 시작... 🚚 (시간이 좀 걸릴 수 있어요!)")

    # 설정에 따라 일부 데이터만 불러올지 결정합니다 (테스트용).
    if config.N_ROWS_TO_LOAD:
        df_full = df_full[:config.N_ROWS_TO_LOAD]

    # --- 1. 기본 데이터 정리 ---
    # 최소 시퀀스 길이를 만족하지 못하는 데이터는 제외합니다.
    df_full = df_full[df_full["sequence_length"] >= min_len_for_split]
    # 모델 학습에 사용하지 않을 컬럼들을 제거합니다.
    df_full = df_full.drop(
        columns=[
            "c0_id", "c1_id", "c2_id", "shipper_name", "shipper_id", 
            "sequence_length", "item_condition_id", "item_condition_name", 
            "size_id", "size_name", "brand_id", "brand_name", 
            "color", "price", "product_id"
        ],
        errors='ignore'  # 해당 컬럼이 없어도 오류를 내지 않음
    )
    
    logger.info("전처리 후 데이터 확인...")
    logger.info(f"데이터 형태: {df_full.shape}")
    logger.info(f"컬럼: {df_full.columns.tolist()}")

    # 'event_id' 컬럼을 'category' 타입으로 변경하여 메모리를 효율적으로 사용합니다.
    df_full["event_id"] = df_full["event_id"].astype("category")
    logger.info(
        f"'event_id'를 범주형으로 변경했어요. 고유한 행동 종류는 {len(df_full['event_id'].cat.categories)}가지 입니다."
    )

    # 학습, 검증, 테스트 데이터를 담을 리스트를 초기화합니다.
    train_samples, valid_samples, test_samples = [], [], []

    # --- 2. 사용자 행동 시퀀스 그룹화 ---
    # 데이터를 사용자, 세션, 시퀀스 ID 및 시간순으로 정렬합니다.
    df_full_sorted = df_full.sort_values(
        by=["user_id", "session_id", "sequence_id", "stime"]
    )
    # 정렬된 데이터를 바탕으로 개별적인 사용자 행동 시퀀스(여정)별로 그룹화합니다.
    grouped_by_sequence = df_full_sorted.groupby(
        ["user_id", "session_id", "sequence_id"]
    )
    logger.info(
        f"총 {len(grouped_by_sequence)}개의 원본 쇼핑 여정(시퀀스) 묶음을 찾았어요. 이제 나눠볼게요..."
    )

    # --- 3. 아이템 및 이벤트 ID 맵핑 생성 ---
    # 각 아이템 ID를 고유한 숫자(인덱스)로 변환하기 위한 사전을 준비합니다.
    # "<PAD_ITEM_ID>"는 시퀀스 길이를 맞추기 위한 가상의 아이템입니다.
    item_id_to_idx = {"<PAD_ITEM_ID>": 0}
    # 이전에 만들어둔 맵핑 파일이 있으면 불러와서 사용합니다.
    if os.path.exists(config.ITEM_ID_IDX_PATH):
        with open(config.ITEM_ID_IDX_PATH, "rb") as f:
            item_id_to_idx = pickle.load(f)

    # 현재 데이터에만 있는 새로운 아이템들을 찾아 맵핑 사전에 추가합니다.
    unique_item_ids = [item_id for item_id in df_full["item_id"].unique().tolist() if item_id not in item_id_to_idx.keys()]
    new_item_id_to_idx = {item_id: len(item_id_to_idx) + i for i, item_id in enumerate(unique_item_ids)}
    
    # 새로운 아이템 정보를 기존 사전에 업데이트하고, 파일로 저장합니다.
    item_id_to_idx.update(new_item_id_to_idx)
    with open(config.ITEM_ID_IDX_PATH, "wb") as f:
        pickle.dump(item_id_to_idx, f)

    # 인덱스에서 아이템 ID로 변환하기 위한 역방향 맵핑을 만듭니다.
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    logger.info(f"총 아이템 개수 (패딩 포함): {len(item_id_to_idx)}")

    # 각 이벤트(행동)를 고유한 숫자로 변환하기 위한 사전을 정의합니다.
    event_to_idx = {
        "<PAD_EVENT>": 0, "item_view": 1, "item_like": 2, "item_add_to_cart_tap": 3,
        "offer_make": 4, "buy_start": 5, "buy_comp": 6,
    }
    logger.info(f"총 이벤트 개수 (패딩 포함): {len(event_to_idx)}")
    
    # --- 4. 아이템 이름 임베딩 생성 및 관리 ---
    # 아이템 인덱스를 해당 아이템의 이름 임베딩(숫자 벡터)에 매핑하는 사전을 준비합니다.
    item_idx_to_embedded_name = {}
    # 이전에 생성한 임베딩 파일이 있으면 불러옵니다.
    if os.path.exists(config.ITEM_IDX_NAME_PATH):
        logger.info(f"기존 임베딩 파일 로드 및 업데이트: {config.ITEM_IDX_NAME_PATH}")
        loaded_data = torch.load(config.ITEM_IDX_NAME_PATH, map_location='cpu', weights_only=False)
        keys = loaded_data["keys"]
        tensors = loaded_data["tensors"]
        item_idx_to_embedded_name = {key: tensors[i] for i, key in enumerate(keys)}
    
    # 데이터에서 고유한 아이템 목록을 추출합니다.
    unique_items_df = df_full[['item_id', 'name']].drop_duplicates(subset=['item_id'])
    
    # 아직 임베딩이 생성되지 않은 새로운 아이템들만 필터링합니다.
    unique_items_df['idx'] = unique_items_df['item_id'].map(item_id_to_idx)
    new_items_df = unique_items_df[~unique_items_df['idx'].isin(item_idx_to_embedded_name.keys())]
    
    # 새로운 아이템이 있다면, 이에 대한 임베딩을 생성합니다.
    if not new_items_df.empty:
        new_items_to_embed = pd.Series(new_items_df['name'].values, index=new_items_df['idx']).to_dict()
        logger.info(f"{len(new_items_to_embed)}개의 새로운 아이템에 대한 임베딩 생성 중...")
        
        new_indices = list(new_items_to_embed.keys())
        new_names = list(new_items_to_embed.values())
        
        # SentenceTransformer 모델을 사용하여 이름(텍스트)을 임베딩(숫자 벡터)으로 변환합니다.
        new_embeddings = sentence_model.encode(
            new_names,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        
        # 새로 생성된 임베딩을 기존 딕셔너리에 추가하고 파일로 저장합니다.
        item_idx_to_embedded_name.update(dict(zip(new_indices, new_embeddings)))

        keys_list = list(item_idx_to_embedded_name.keys())
        tensor_list = list(item_idx_to_embedded_name.values())
        data_to_save = {"keys": keys_list, "tensors": tensor_list}
        torch.save(data_to_save, config.ITEM_IDX_NAME_PATH)
        logger.info("임베딩 파일 업데이트 완료.")

    # --- 4.5. 클래스 가중치 계산 (카테고리 불균형 해소) ---
    logger.info("대분류 카테고리(c0_name) 빈도 기반 가중 손실을 위한 가중치 계산 시작...")
    
    class_weights = None
    if 'c0_name' in df_full.columns:
        # 1. 아이템별 카테고리 정보 매핑 생성
        # drop_duplicates: 아이템 ID당 하나의 c0_name만 있도록 보장합니다. (데이터에 오류가 있을 경우 대비)
        item_id_to_c0_name = df_full[['item_id', 'c0_name']].drop_duplicates(subset=['item_id']).set_index('item_id')['c0_name']

        # 2. 카테고리별 빈도수 계산
        # 전체 데이터셋의 카테고리 분포를 사용합니다.
        category_counts = df_full['c0_name'].value_counts()
        
        # 3. 역빈도 가중치 계산 (Inverse Frequency Weighting)
        # 공식: weight = 전체 샘플 수 / (클래스 수 * 클래스별 샘플 수)
        # 이를 통해 소수 클래스(빈도가 낮은 카테고리)에 더 높은 가중치를 부여합니다.
        total_samples = category_counts.sum()
        num_categories = len(category_counts)
        
        # category_counts가 비어있는 경우를 대비
        if num_categories > 0:
            category_weight_map = (total_samples / (num_categories * category_counts)).to_dict()
        else:
            category_weight_map = {}

        # 4. 아이템 인덱스별 가중치 텐서 생성
        num_items = len(item_id_to_idx)
        class_weights_list = [1.0] * num_items # 기본 가중치는 1로 설정
        
        for item_id, idx in item_id_to_idx.items():
            if item_id == "<PAD_ITEM_ID>":
                class_weights_list[idx] = 0.0 # PAD 토큰은 손실 계산에서 제외
                continue
            
            # 아이템 ID에 해당하는 카테고리 이름을 찾고, 그 카테고리의 가중치를 할당
            c0_name = item_id_to_c0_name.get(item_id)
            if c0_name and c0_name in category_weight_map:
                class_weights_list[idx] = category_weight_map[c0_name]

        class_weights = torch.FloatTensor(class_weights_list)
        
        logger.info("카테고리 기반 가중치 계산 완료.")
        if category_weight_map:
            logger.info(f"계산된 카테고리 가중치 (상위 5개 샘플): { {k: v for k, v in list(category_weight_map.items())[:5]} }")
    else:
        logger.warning("'c0_name' 컬럼을 찾을 수 없어 가중치를 계산할 수 없습니다. 모든 아이템 가중치를 1로 설정합니다.")
        num_items = len(item_id_to_idx)
        class_weights = torch.ones(num_items)
        class_weights[item_id_to_idx["<PAD_ITEM_ID>"]] = 0.0

    # --- 5. 시퀀스 분할 및 학습 데이터 생성 ---
    logger.info("사용자(세션) 기반으로 Train/Valid/Test 데이터 분할 시작...")
    
    # 전체 시퀀스 그룹을 리스트로 변환
    all_sequences = list(grouped_by_sequence)
    # 재생산성을 위해 랜덤 시드 고정
    random.seed(42)
    # 그룹을 무작위로 섞음
    random.shuffle(all_sequences)

    # 데이터셋 크기 계산
    total_size = len(all_sequences)
    train_size = int(total_size * 0.7)
    valid_size = int(total_size * 0.2)
    
    # 그룹 분할
    train_groups = all_sequences[:train_size]
    valid_groups = all_sequences[train_size : train_size + valid_size]
    test_groups = all_sequences[train_size + valid_size :]

    logger.info(f"데이터 그룹 분할 완료: Train: {len(train_groups)}, Valid: {len(valid_groups)}, Test: {len(test_groups)} 그룹")

    def create_samples_from_groups(groups, dataset_type):
        """
        주어진 그룹으로부터 학습/검증/테스트 샘플을 생성하는 함수.
        Leave-one-out 방식을 각 그룹 내에서만 적용.
        """
        samples = []
        for _, group in tqdm(groups, desc=f"{dataset_type} 샘플 생성 중"):
            item_ids = group["item_id"].tolist()
            
            name_sequences = [
                item_idx_to_embedded_name.get(item_id_to_idx.get(item_id))
                for item_id in item_ids
            ]
            
            event_sequences = [
                event_to_idx.get(e, 0) for e in group["event_id"].tolist()
            ]

            item_idx_sequences = [
                item_id_to_idx[item_id] for item_id in item_ids
            ]

            current_paired_sequences = list(
                zip(name_sequences, event_sequences, item_idx_sequences)
            )

            if len(current_paired_sequences) < config.MIN_LEN_FOR_SEQ_SPLIT:
                continue

            # --- Leave-One-Out 샘플링 ---
            # 각 사용자 시퀀스 내에서, 마지막 상호작용을 예측하는 방식으로 데이터를 구성합니다.
            # 이 방식은 이제 각 분할된 데이터셋(Train/Valid/Test) 그룹 내에서만 독립적으로 적용됩니다.
            # 예시 시퀀스: [A, B, C, D, E]
            if dataset_type == 'test':
                # 테스트셋: 사용자의 가장 마지막 행동을 예측합니다.
                # Input: [A, B, C, D] -> Target: E
                input_sequences = current_paired_sequences[:-1]
                target_item_idx = current_paired_sequences[-1][2]
                if input_sequences:
                    samples.append(
                        (
                            (
                                [s[0] for s in input_sequences],
                                [s[1] for s in input_sequences],
                            ),
                            target_item_idx,
                        )
                    )
            elif dataset_type == 'valid':
                # 검증셋: 마지막에서 두 번째 행동을 예측합니다.
                # 이를 통해 학습 중 모델의 일반화 성능을 평가합니다.
                # Input: [A, B, C] -> Target: D
                input_sequences = current_paired_sequences[:-2]
                target_item_idx = current_paired_sequences[-2][2]
                if input_sequences:
                    samples.append(
                        (
                            (
                                [s[0] for s in input_sequences],
                                [s[1] for s in input_sequences],
                            ),
                            target_item_idx,
                        )
                    )
            else: # train
                # 학습셋: 시퀀스를 점진적으로 늘려가며 여러 학습 샘플을 만듭니다.
                # 이를 통해 모델이 시퀀스의 다음 아이템을 예측하는 패턴을 학습합니다.
                # 예시:
                # Input: [A]       -> Target: B
                # Input: [A, B]    -> Target: C
                train_sequences_all = current_paired_sequences[:-2]
                for i in range(1, len(train_sequences_all)):
                    train_input = train_sequences_all[:i]
                    train_target = train_sequences_all[i][2]
                    if train_input:
                        samples.append(
                            (
                                (
                                    [s[0] for s in train_input],
                                    [s[1] for s in train_input],
                                ),
                                train_target,
                            )
                        )
        return samples

    # 각 그룹으로부터 샘플 생성
    train_samples = create_samples_from_groups(train_groups, 'train')
    valid_samples = create_samples_from_groups(valid_groups, 'valid')
    test_samples = create_samples_from_groups(test_groups, 'test')

    # --- 6. 최종 정리 ---
    # 추천 결과를 보여줄 때 사용할 아이템 정보를 담은 데이터프레임을 만듭니다.
    df_item_info = None
    if "item_id" in df_full.columns and "name" in df_full.columns:
        df_item_info = (
            df_full[["item_id", "name", "c0_name", "c1_name", "c2_name"]]
                .drop_duplicates(subset=["item_id"])
                .set_index("item_id")
        )
        logger.info(f"추천 결과에 표시할 아이템 이름 정보 {len(df_item_info)}개를 준비했어요.")

    logger.info(f"데이터 분할 완료: Train {len(train_samples)}, Valid {len(valid_samples)}, Test {len(test_samples)} 샘플")
    
    end_time = time.time()
    logger.info(f"데이터 전처리 완료! 총 {end_time - start_time:.2f}초 걸렸어요.")

    # 메모리 정리를 위해 더 이상 필요 없는 모델 객체를 삭제합니다.
    del sentence_model
    gc.collect()

    # 최종적으로 생성된 데이터들을 반환합니다.
    return (
        train_samples, valid_samples, test_samples,
        item_id_to_idx, event_to_idx, idx_to_item_id,
        item_idx_to_embedded_name, df_item_info, class_weights
    ) 