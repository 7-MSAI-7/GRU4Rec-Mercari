# -*- coding: utf-8 -*-
"""
이 파일은 프로젝트 전체에서 사용되는 다양한 설정값들을 모아두는 곳입니다.
데이터 파일의 위치, 모델의 구조, 학습 파라미터 등 중요한 값들을 여기서 관리합니다.
이 파일을 수정하면 코드의 다른 부분을 변경하지 않고도 실험 설정을 쉽게 바꿀 수 있습니다.
"""
import os
import torch

# 💻 --- 장치 설정 ---
# 모델을 CPU에서 실행할지, GPU(그래픽카드)에서 실행할지 자동으로 결정합니다.
# GPU(cuda)가 있으면 훨씬 빠르게 학습할 수 있습니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📂 --- 데이터 및 로그 경로 설정 ---
# Mercari 쇼핑몰 데이터 파일이 있는 위치를 지정합니다.
DATA_PATH = r"/home/j1star/datasets/merrec/20230501/00000000000[0-4].parquet"

# 로그(프로그램 실행 기록) 파일을 저장할 디렉터리입니다.
LOG_FILE_DIR = "logs"

# 🗃️ --- 모델 관련 파일 경로 설정 ---
# 학습된 모델이나 데이터 처리 중 생성되는 파일들을 저장할 디렉터리입니다.
MODEL_ARTIFACTS_DIR = "model_artifacts"
# 최종 학습된 모델의 파라미터가 저장될 경로입니다.
MODEL_SAVE_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "lastest_gru_recommender.pth")
# 아이템 인덱스와 임베딩된 이름(숫자 벡터)을 매핑한 파일 경로입니다.
ITEM_IDX_NAME_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "idx_to_embedded_name.pt")
# 아이템 ID와 내부 인덱스를 매핑한 파일 경로입니다.
ITEM_ID_IDX_PATH = os.path.join(MODEL_ARTIFACTS_DIR, "item_id_to_idx.pkl")

# 📊 --- 데이터 처리 및 분할 관련 설정 ---
# 컴퓨터 메모리가 부족할 때, 전체 데이터 대신 일부만 불러와서 빠르게 테스트해볼 수 있습니다.
# (None으로 설정하면 전체 데이터를 사용합니다.)
N_ROWS_TO_LOAD = None
# 🚶‍♂️🚶‍♀️🚶 사용자의 쇼핑 기록(시퀀스)이 최소 이 길이 이상이어야 학습/검증/테스트용으로 의미가 있다고 판단합니다.
MIN_LEN_FOR_SEQ_SPLIT = 3

# 🧠 --- GRU 모델 구조 설정 ---
# 🎁 각 상품의 이름(텍스트)을 얼마나 자세한 숫자 벡터로 표현할지 결정합니다. 숫자가 클수록 더 자세하지만 계산이 복잡해집니다.
NAME_EMBEDDING_DIM = 384
# ✨ 각 행동(예: '상품보기', '좋아요')을 얼마나 자세한 숫자 벡터로 표현할지 결정합니다.
EVENT_EMBEDDING_DIM = 32
# 🧠 GRU 모델이 한 번에 얼마나 많은 정보를 기억할지(기억 용량) 결정합니다. 높을수록 복잡한 패턴을 학습할 수 있습니다.
GRU_HIDDEN_DIM = 128
# 🏢 GRU 층을 몇 개나 쌓을지 결정합니다. 깊을수록 더 복잡한 관계를 학습할 수 있지만 과적합의 위험이 있습니다.
GRU_NUM_LAYERS = 2
# 💧 학습 시 모델의 일부 연결을 무작위로 끊어서, 모델이 학습 데이터에만 너무 의존하지 않도록(과적합 방지) 합니다.
DROPOUT_RATE = 0.5

# 📦 --- 데이터 로더 설정 ---
# 한 번에 모델에 몇 개의 데이터를 묶어서 보여줄지 결정합니다. (학습의 안정성과 속도에 영향을 줍니다.)
BATCH_SIZE = 256

# ⚙️ --- 모델 학습 관련 하이퍼파라미터 ---
# 전체 데이터셋을 총 몇 번 반복해서 학습할지 결정합니다.
N_EPOCHS = 40
# 학습률: 모델이 정답을 향해 얼마나 큰 보폭으로 나아갈지 결정합니다.
LEARNING_RATE = 0.0005
# 가중치 감소: 모델의 파라미터(가중치)가 너무 커지는 것을 방지하여 과적합을 막는 기술입니다.
WEIGHT_DECAY = 0.001
# 그래디언트 클리핑: 학습 과정에서 모델의 파라미터가 비정상적으로 크게 업데이트되는 것을 방지합니다.
CLIP_GRAD_NORM = 1.0

# 그래디언트 축적 (Gradient Accumulation)
# 몇 번의 미니배치(mini-batch)마다 그래디언트를 업데이트할지 결정합니다.
# 예를 들어, BATCH_SIZE가 128이고 ACCUMULATION_STEPS가 4이면,
# 실질적인 배치 사이즈는 128 * 4 = 512가 됩니다.
ACCUMULATION_STEPS = 4

# 🎯 --- 추론 및 평가 관련 설정 ---
# 추천을 생성할 때, 상위 몇 개의 아이템을 보여줄지 결정합니다.
TOP_N = 20
# 모델의 성능을 평가할 때, 상위 몇 개의 추천 결과 안에 정답이 포함되어 있는지를 기준으로 삼을지 결정합니다.
K_FOR_METRICS = 20
