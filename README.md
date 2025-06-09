# MERRec: 세션 기반 추천 시스템

이 프로젝트는 사용자의 최근 행동 시퀀스(세션)를 기반으로 다음에 관심을 가질 만한 상품을 추천하는 딥러닝 기반 추천 시스템입니다. 사용자가 조회하거나, '좋아요'를 누르거나, 장바구니에 담는 등의 행동 패턴을 학습하여 개인화된 추천을 제공하는 것을 목표로 합니다.

## 🌟 주요 특징

- **세션 기반 추천**: 사용자의 단기적인 관심사를 파악하여 실시간으로 변화하는 선호도에 맞는 추천을 제공합니다.
- **GRU 모델 활용**: 순차적인 데이터(사용자 행동 로그)의 특징을 효과적으로 학습하기 위해 GRU(Gated Recurrent Unit) 딥러닝 모델을 사용합니다.
- **풍부한 컨텍스트 활용**: 상품의 이름(Text) 정보와 사용자의 행동(Event) 정보를 함께 임베딩하여 모델의 예측 성능을 높입니다.
- **모듈화된 구조**: 데이터 처리, 모델, 학습 엔진 등 기능별로 코드가 모듈화되어 있어 유지보수와 확장이 용이합니다.

## 📊 데이터셋

이 프로젝트는 **MerRec (mercari-us/merrec)** 데이터셋을 사용하여 학습되었습니다. MerRec은 C2C 마켓플레이스 이커머스 플랫폼인 Mercari의 상품 상호작용 이벤트 시퀀스 데이터를 대규모로 수집하여 익명화한 데이터셋입니다.

-   **데이터셋 주소**: [https://huggingface.co/datasets/mercari-us/merrec](https://huggingface.co/datasets/mercari-us/merrec)
-   **특징**:
    -   **대용량**: 5백만 명 이상의 고유 사용자, 8천만 개 이상의 고유 아이템, 10억 개 이상의 이벤트 로그를 포함합니다.
    -   **다양성**: C2C(소비자 간 거래) 환경의 풍부한 상품 피처와 사용자 행동 패턴을 연구하는 데 적합합니다.
    -   **형식**: 데이터는 Parquet 형식으로 제공됩니다.

## 🗂️ 프로젝트 구조

```
.
├── .venv/                   # 가상 환경
├── logs/                    # 로그 파일 저장 디렉토리
├── model_artifacts/         # 학습된 모델 및 관련 데이터 저장 디렉토리
├── src/                     # 소스 코드 디렉토리
│   ├── datasets/            # PyTorch Dataset 관련 클래스
│   ├── models/              # 모델 아키텍처 (GRU)
│   ├── __pycache__/
│   ├── data_processing.py   # 데이터 로드 및 전처리
│   ├── logger_config.py     # 로깅 설정
│   ├── settings.py          # 프로젝트 전역 설정 (경로, 하이퍼파라미터 등)
│   ├── training_engine.py   # 모델 학습 및 평가 로직
│   └── utils.py             # 유틸리티 함수
├── README.md                # 프로젝트 설명 파일
├── requirements.txt         # 프로젝트 의존성 패키지 목록
├── train.py                 # 모델 학습 스크립트
└── predict.py               # 학습된 모델을 사용한 추천 생성 스크립트
```

## ⚙️ 설치 및 환경 설정

### 1. Python 버전

이 프로젝트는 **Python 3.12.9**를 기준으로 개발되었습니다. 원활한 실행을 위해 해당 버전 또는 호환되는 버전의 Python을 준비해주세요.

### 2. 저장소 복제

```bash
git clone https://github.com/your-username/merrec_recommendation_system.git
cd merrec_recommendation_system
```

### 3. 가상 환경 및 의존성 설치

프로젝트의 독립적인 환경을 위해 가상 환경 사용을 권장합니다.

```bash
# 가상 환경 생성
python -m venv .venv

# 가상 환경 활성화 (Windows)
.\.venv\Scripts\activate

# 가상 환경 활성화 (macOS/Linux)
source .venv/bin/activate

# 의존성 패키지 설치
pip install -r requirements.txt
```
*참고: CUDA GPU를 사용하여 학습을 진행한다면 `torch` 관련 패키지는 사용자의 CUDA 버전에 맞게 직접 설치해야 할 수 있습니다.*
```bash
# 예시: CUDA 12.8 환경
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 🚀 사용법

### 1. 데이터 준비

-   모델을 학습시키기 위한 사용자 행동 로그 데이터가 필요합니다. `Parquet` 형식의 파일을 `data/` 디렉토리(또는 `src/settings.py`에 지정된 경로)에 위치시켜야 합니다.
-   `src/settings.py` 파일에서 `DATA_PATH` 변수를 실제 데이터 경로에 맞게 수정해주세요.

### 2. 모델 학습

-   다음 명령어를 실행하여 모델 학습을 시작합니다. 학습 과정, 검증 결과, 최종 평가 지표가 로그로 출력됩니다.
    ```bash
    python train.py
    ```
-   학습이 완료되면 최적의 모델 가중치가 `model_artifacts/lastest_gru_model.pth` 경로에 저장됩니다.

### 3. 추천 생성

-   학습된 모델을 사용하여 새로운 추천을 생성하려면 다음 명령어를 실행합니다.
    ```bash
    python predict.py
    ```
-   스크립트는 임의의 사용자 행동 시퀀스를 생성하고, 이를 바탕으로 상위 N개의 추천 상품 목록을 출력합니다.

## 🤖 모델

### 아키텍처
이 시스템은 **GRU(Gated Recurrent Unit)** 네트워크를 기반으로 합니다.

**GRU(Gated Recurrent Unit)**는 순환 신경망(RNN)의 한 종류로, 시계열 데이터나 순차적 데이터의 패턴을 학습하는 데 효과적입니다. 특히 기존 RNN의 장기 의존성 문제(오래전 정보를 기억하지 못하는 문제)를 해결하기 위해 설계되었습니다. GRU는 '업데이트 게이트'와 '리셋 게이트'라는 두 가지 게이트를 사용하여 시퀀스 내에서 어떤 정보를 기억하고, 어떤 정보를 잊을지를 결정합니다. 이 특성 덕분에 사용자의 행동 흐름 속에서 중요한 맥락을 포착하여 다음 행동을 예측하는 이번 추천 시스템에 매우 적합합니다.

1.  **입력**: 사용자의 행동 시퀀스가 모델의 입력으로 사용됩니다. 각 시점의 입력은 (상품명 임베딩, 행동 타입) 쌍으로 구성됩니다.
2.  **임베딩**:
    -   **상품명**: `Sentence-Transformer`를 사용해 상품명을 의미론적 벡터(Semantic Vector)로 변환합니다.
    -   **행동**: '조회', '좋아요', '구매시작' 등 사용자의 행동 유형은 학습 가능한 임베딩 레이어를 통해 벡터로 변환됩니다.
3.  **GRU 레이어**: 두 임베딩이 결합된 벡터 시퀀스가 GRU 레이어를 통과하며, 시퀀스의 문맥 정보가 압축된 '생각 벡터(Thought Vector)'를 출력합니다.
4.  **출력**: GRU의 최종 은닉 상태(Hidden State)가 선형 레이어(Fully-Connected Layer)를 거쳐 전체 상품에 대한 선호도 점수(Logits)를 계산하고, 가장 높은 점수를 가진 상품을 다음 아이템으로 추천합니다.

### 성능 측정
-   모델 학습 중 Recall@K, MRR@K 등의 지표를 통해 성능을 모니터링합니다.

## 💡 개선 방향

-   **다양한 모델 실험**: GRU 외에 Transformer, BERT4Rec 등 최신 시퀀스 모델을 적용하여 성능을 비교할 수 있습니다.
-   **피처 엔지니어링**: 상품의 카테고리, 가격, 이미지 등 더 다양한 피처를 활용하여 모델의 예측력을 높일 수 있습니다.
-   **서빙 파이프라인 구축**: 학습된 모델을 실시간으로 서비스할 수 있도록 FastAPI, Flask 등을 이용한 API 서버를 구축하고 MLOps 파이프라인을 구성할 수 있습니다.
