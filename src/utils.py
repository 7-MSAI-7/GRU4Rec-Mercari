# -*- coding: utf-8 -*-
"""
이 파일은 프로젝트 전반에서 사용되는 보조적인 유틸리티 함수들을 담고 있습니다.
- `transfer_weights`: 기존에 학습된 모델의 가중치를 새로운 모델 구조에 맞게 옮겨주는 역할을 합니다.
- `collate_fn`: 데이터 로더가 모델에 데이터를 전달하기 전에, 데이터를 일정한 형태로 정리하고 묶어주는 역할을 합니다.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any

def transfer_weights(old_state_dict: Dict[str, Any], new_model: torch.nn.Module) -> Dict[str, Any]:
    """
    기존 모델(old_state_dict)의 가중치(학습된 파라미터)를 새로운 모델(new_model)로 이전합니다.
    예를 들어, 새로운 아이템이 추가되어 모델의 최종 출력층 크기가 바뀌었을 때,
    바뀌지 않은 다른 부분의 가중치는 그대로 가져와서 처음부터 다시 학습할 필요가 없도록 도와줍니다.

    Args:
        old_state_dict (Dict[str, Any]): 이전에 저장된 모델의 상태 사전.
        new_model (torch.nn.Module): 가중치를 전달받을 새로운 모델.

    Returns:
        Dict[str, Any]: 이전 가중치가 적용된 새로운 모델의 상태 사전.
    """
    # 새로운 모델의 파라미터 구조를 가져옵니다.
    new_state_dict = new_model.state_dict()
    
    # 이전 모델이 알고 있던 아이템의 개수를 확인합니다.
    old_num_items = old_state_dict["fc.weight"].size(0)

    print("기존 모델의 가중치를 새로운 모델로 이전합니다...")
    # 이전 모델의 각 파라미터(이름, 값)에 대해 반복합니다.
    for name, param in old_state_dict.items():
        # 파라미터 이름이 새로운 모델에도 존재하는 경우
        if name in new_state_dict:
            # 파라미터의 모양(크기)이 완전히 동일하면, 그대로 복사합니다.
            if new_state_dict[name].shape == param.shape:
                new_state_dict[name].copy_(param)
            # 만약 파라미터가 최종 출력층의 가중치('fc.weight')이고 모양이 다르다면 (아이템 개수가 바뀐 경우)
            elif name == "fc.weight":
                print(
                    f"  - 출력 레이어(weight) 이전 중... (이전 아이템 수: {old_num_items})"
                )
                # 공통된 부분(이전 아이템 개수만큼)만 가중치를 복사합니다.
                new_state_dict[name][:old_num_items, :].copy_(param)
            # 최종 출력층의 편향('fc.bias')이고 모양이 다르다면
            elif name == "fc.bias":
                print(
                    f"  - 출력 레이어(bias) 이전 중... (이전 아이템 수: {old_num_items})"
                )
                # 공통된 부분(이전 아이템 개수만큼)만 편향을 복사합니다.
                new_state_dict[name][:old_num_items].copy_(param)

    # 새로운 모델에 이전 가중치가 적용된 상태를 불러옵니다.
    new_model.load_state_dict(new_state_dict)
    print("가중치 이전 완료.")
    
    return new_state_dict

def collate_fn(batch):
    """
    PyTorch의 DataLoader가 데이터를 배치(batch) 단위로 묶을 때 사용하는 사용자 정의 함수입니다.
    각 사용자의 행동 시퀀스는 길이가 제각각 다르기 때문에,
    이를 하나의 묶음(배치)으로 만들기 위해 가장 긴 시퀀스 길이에 맞춰 나머지 시퀀스들의 길이를 늘려주는 '패딩(padding)' 작업을 수행합니다.

    Args:
        batch (list): 데이터셋에서 꺼내온 샘플들의 리스트. 각 샘플은 (입력 시퀀스, 정답 아이템) 형태입니다.

    Returns:
        tuple: 패딩 처리된 입력 텐서 묶음과 정답 텐서를 반환합니다.
    """
    # 배치로부터 입력 시퀀스들과 정답 아이템들을 분리합니다.
    input_sequences, target_items = zip(*batch)
    # 입력 시퀀스에서 다시 이름 임베딩 시퀀스와 이벤트 ID 시퀀스를 분리합니다.
    name_embed_sequences, event_id_sequences = zip(*input_sequences)

    # 각 시퀀스를 PyTorch 텐서로 변환합니다. (현재는 이미 텐서로 가정하고 리스트로만 받음)
    name_embed_tensors = list(name_embed_sequences)
    event_id_tensors = list(event_id_sequences)
    
    # `pad_sequence` 함수를 사용하여 시퀀스들의 길이를 맞춥니다.
    # `batch_first=True`는 텐서의 차원을 (배치 크기, 시퀀스 길이, 특징) 순서로 만듭니다.
    # `padding_value=0`은 부족한 부분을 0으로 채우라는 의미입니다.
    padded_name_embeds = pad_sequence(name_embed_tensors, batch_first=True, padding_value=0)
    padded_event_ids = pad_sequence(event_id_tensors, batch_first=True, padding_value=0)
    
    # 정답 아이템들도 하나의 텐서로 결합합니다.
    targets = torch.tensor(target_items, dtype=torch.long)

    # 최종적으로 모델에 입력될 형태로 데이터를 반환합니다.
    return (padded_name_embeds, padded_event_ids), targets