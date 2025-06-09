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

    # --- 5. 시퀀스 분할 및 학습 데이터 생성 ---
    # 그룹화된 각 사용자 행동 시퀀스에 대해 반복 작업을 수행합니다.
    for _, group in tqdm(grouped_by_sequence, desc="시퀀스 분할 및 샘플링"):
        item_ids = group["item_id"].tolist()
        
        # 현재 시퀀스에 포함된 아이템들의 이름 임베딩을 가져옵니다.
        name_sequences = [
            item_idx_to_embedded_name.get(item_id_to_idx.get(item_id))
            for item_id in item_ids
        ]
        
        # 현재 시퀀스에 포함된 이벤트들을 인덱스로 변환합니다.
        event_sequences = [
            event_to_idx.get(e, 0) for e in group["event_id"].tolist()
        ]

        # 현재 시퀀스에 포함된 아이템들을 인덱스로 변환합니다.
        item_idx_sequences = [
            item_id_to_idx[item_id] for item_id in item_ids
        ]

        # (이름 임베딩, 이벤트 인덱스, 아이템 인덱스)를 하나의 튜플로 묶습니다.
        current_paired_sequences = list(
            zip(name_sequences, event_sequences, item_idx_sequences)
        )

        # 시퀀스 길이가 너무 짧으면 학습 데이터로 사용하기 부적합하므로 건너뜁니다.
        if len(current_paired_sequences) < config.MIN_LEN_FOR_SEQ_SPLIT:
            continue

        # --- 데이터 분할: Leave-One-Out 방식 ---
        # 사용자의 마지막 행동을 예측하는 방식으로 데이터를 분할합니다.
        # 예: 시퀀스 [A, B, C, D]
        # Test:  Input: [A, B, C] -> Target: D (가장 마지막 행동 예측)
        # Valid: Input: [A, B]    -> Target: C (마지막에서 두 번째 행동 예측)
        # Train: Input: [A]       -> Target: B (그 이전의 모든 상호작용을 학습)

        # 테스트 데이터 생성
        test_input_sequences = current_paired_sequences[:-1]  # 마지막 아이템을 제외한 모든 시퀀스
        test_target_item_idx = current_paired_sequences[-1][2]  # 마지막 아이템의 인덱스가 정답
        if test_input_sequences:
            test_samples.append(
                (
                    (
                        [s[0] for s in test_input_sequences], # 이름 임베딩 시퀀스
                        [s[1] for s in test_input_sequences], # 이벤트 인덱스 시퀀스
                    ),
                    test_target_item_idx, # 정답 아이템 인덱스
                )
            )

        # 검증 데이터 생성
        valid_input_sequences = current_paired_sequences[:-2] # 마지막 두 개 아이템을 제외한 시퀀스
        valid_target_item_idx = current_paired_sequences[-2][2] # 마지막에서 두 번째 아이템이 정답
        if valid_input_sequences:
            valid_samples.append(
                (
                    (
                        [s[0] for s in valid_input_sequences],
                        [s[1] for s in valid_input_sequences],
                    ),
                    valid_target_item_idx,
                )
            )

        # 학습 데이터 생성
        # 시퀀스를 점진적으로 늘려가며 여러 개의 학습 샘플을 만듭니다.
        # 예: [A,B,C]가 있다면, ([A])->B, ([A,B])->C 를 두 개의 학습 샘플로 생성합니다.
        train_sequences = current_paired_sequences[:-2]
        for i in range(1, len(train_sequences)):
            train_input = train_sequences[:i]
            train_target = train_sequences[i][2]
            train_samples.append(
                (
                    (
                        [s[0] for s in train_input],
                        [s[1] for s in train_input],
                    ),
                    train_target,
                )
            )

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
        item_idx_to_embedded_name, df_item_info
    ) 