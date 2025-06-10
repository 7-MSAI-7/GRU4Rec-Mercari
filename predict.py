# -*- coding: utf-8 -*-
"""
이 파일은 학습된 추천 모델을 사용하여 새로운 추천을 생성하는 '추론(inference)' 과정을 수행합니다.
학습된 모델과 관련 데이터를 불러온 뒤, 특정 사용자 행동 시퀀스를 입력으로 받아
다음에 올 만한 아이템을 예측하여 추천 목록을 생성합니다.
"""
import os
import pickle
import random
import glob
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Local Imports ---
# 다른 파일에 정의된 함수나 설정을 불러옵니다.
import src.settings as config  # 설정값을 담고 있는 파일
from src.models.gru_model import GruModel  # 학습된 모델과 동일한 구조의 모델을 불러오기 위함
from src.training_engine import generate_recommendations  # 추천을 생성하는 함수


# SentenceTransformer 모델은 텍스트(예: 상품명)를 컴퓨터가 이해할 수 있는 숫자 벡터(임베딩)로 변환하는 모델입니다.
# 프로그램 시작 시 한 번만 로드하여 계속 사용합니다.
sentence_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_inference_dependencies():
    """
    추론(예측)에 필요한 모델과 데이터를 불러오는 함수입니다.
    학습 과정에서 저장된 모델 가중치와 아이템 정보를 불러와 추론 준비를 합니다.
    """
    print("추론에 필요한 파일을 로드합니다...")

    # 아이템 ID와 내부 인덱스를 매핑하는 사전을 불러옵니다.
    # 이를 통해 모델이 이해하는 인덱스를 실제 아이템 ID로 변환할 수 있습니다.
    if os.path.exists(config.ITEM_ID_IDX_PATH):
        with open(config.ITEM_ID_IDX_PATH, "rb") as f:
            item_id_to_idx = pickle.load(f)
        idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    else:
        # 파일이 없으면 오류를 발생시켜 프로그램을 중단합니다.
        raise FileNotFoundError(f"{config.ITEM_ID_IDX_PATH} 파일을 찾을 수 없습니다.")
    
    print(f"학습 모델 로드 중...")
    if os.path.exists(config.MODEL_SAVE_PATH):
        model_state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    else:
        raise FileNotFoundError(f"{config.MODEL_SAVE_PATH} 파일을 찾을 수 없습니다.")

    # 이벤트(사용자 행동)와 내부 인덱스를 매핑하는 사전을 정의합니다.
    # 학습 때 사용했던 것과 동일한 구조여야 합니다.
    event_to_idx = {
        "<PAD_EVENT>": 0,  # 시퀀스 길이를 맞추기 위한 가짜 이벤트
        "item_view": 1,  # 아이템 조회
        "item_like": 2,  # '좋아요'
        "item_add_to_cart_tap": 3,  # 장바구니에 담기
        "offer_make": 4,  # 가격 제안
        "buy_start": 5,  # 구매 시작
        "buy_comp": 6,  # 구매 완료
    }

    # 모델을 초기화하고 학습된 가중치를 로드할 준비를 합니다.
    # 학습 때와 동일한 파라미터로 모델 구조를 만들어야 합니다.
    print(f"모델 초기화 중...")
    model_args = {
        "device": config.DEVICE,
        "name_embedding_dim": config.NAME_EMBEDDING_DIM,
        "event_embedding_dim": config.EVENT_EMBEDDING_DIM,
        "gru_hidden_dim": config.GRU_HIDDEN_DIM,
        "gru_num_layers": config.GRU_NUM_LAYERS,
        "dropout_rate": config.DROPOUT_RATE,
        "n_events": len(event_to_idx),  # 전체 이벤트 종류 수
        "n_items": len(item_id_to_idx),  # 전체 아이템 종류 수
    }
    model = GruModel(**model_args)
    model.load_state_dict(model_state_dict)

    # 모델을 추론 모드(eval)로 설정합니다.
    # 이는 학습 때와 달리 드롭아웃 등의 기능을 비활성화하여 일관된 예측을 하도록 합니다.
    model.to(config.DEVICE)
    model.eval()

    print("로드 완료.")
    # 준비된 모델과 데이터(사전)를 반환합니다.
    return model, idx_to_item_id, item_id_to_idx


if __name__ == "__main__":
    # 이 스크립트가 직접 실행될 때 아래 코드가 실행됩니다.
    
    # 추론에 필요한 모델과 데이터를 불러옵니다.
    model, idx_to_item_id, item_id_to_idx = load_inference_dependencies()

    # 추천의 기반이 될 아이템 정보 데이터프레임을 불러옵니다.
    # df_item_info = pd.read_parquet(
    #     r"D:/Downloads/merrec/20230501/000000000000.parquet"
    # )
    parquet_files = glob.glob(config.DATA_PATH)
    df_item_info = pd.concat([pd.read_parquet(file) for file in parquet_files])

    # 아이템 ID를 기준으로 중복을 제거하고 필요한 컬럼만 선택합니다.
    df_item_info = (
        df_item_info[["item_id", "name", "c0_name", "c1_name", "c2_name"]]
        .drop_duplicates(subset=["item_id"])
        .set_index("item_id")
    )

    # 이벤트 맵핑 사전을 다시 정의합니다. (load_inference_dependencies 함수 내의 것과 동일)
    event_to_idx = {
        "<PAD_EVENT>": 0,
        "item_view": 1,
        "item_like": 2,
        "item_add_to_cart_tap": 3,
        "offer_make": 4,
        "buy_start": 5,
        "buy_comp": 6,
    }

    # --- 추천 생성을 위한 가상 시나리오 ---
    print("추천 결과 생성 예시... 🛍️")
        
    # 아이템 정보에서 무작위로 1~10개의 아이템-이벤트 시퀀스를 샘플링하여 가상 사용자 행동을 만듭니다.
    item_event_sequences = df_item_info.sample(n=random.randint(1, 10))
    item_event_sequences = item_event_sequences.apply(
        lambda x: (
            x["name"],  # 상품명
            random.choice(list(event_to_idx.keys())[1:]),  # 랜덤 행동
            x["c0_name"],  # 대분류 카테고리
            x["c1_name"],  # 중분류 카테고리
            x["c2_name"],  # 소분류 카테고리
        ),
        axis=1,
    ).tolist()
    
    # 모델에 입력으로 사용할 시퀀스 정보를 출력합니다.
    print(f"추천 생성을 위한 입력 시퀀스:")
    for item_event_sequence in item_event_sequences:
        print(
            f"  - 상품명: {item_event_sequence[0]:>80} | 행동: {item_event_sequence[1]:>20}"
            f"  - 카테고리: {(item_event_sequence[2] if item_event_sequence[2] else ' '):<20} | {(item_event_sequence[3] if item_event_sequence[3] else ' '):<20} | {(item_event_sequence[4] if item_event_sequence[4] else ' '):<20}"
        )

    # 준비된 모델과 입력 시퀀스를 사용하여 추천을 생성합니다.
    recommendations_df = generate_recommendations(
        model=model,
        item_event_sequences=item_event_sequences,  # 입력 시퀀스
        top_n=config.TOP_N,  # 상위 몇 개를 추천할지
        device=config.DEVICE,
        idx_to_item_id=idx_to_item_id,
        df_item_info=df_item_info,
        event_to_idx=event_to_idx
    )

    # 생성된 추천 결과를 확인하고 출력합니다.
    if not recommendations_df.empty:
        print(
            f"\n🎁 다음으로 이 아이템들은 어떠세요? (상위 {config.TOP_N}개 추천 결과)"
        )
        print(recommendations_df.to_string())
    else:
        print("죄송해요, 지금은 추천해드릴 만한 아이템을 찾지 못했어요. 😔")

