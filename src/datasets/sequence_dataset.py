import torch
from torch.utils.data import Dataset
import random

class SequenceDataset(Dataset):
    def __init__(self, paired_sequences, is_train=False, augmentation_prob=0.2):
        """
        초기화 메서드
        - paired_sequences: ((name_embeddings, event_ids), target_item_id) 형태의 튜플 리스트
        - is_train (bool): True이면 데이터 증강을 적용합니다.
        - augmentation_prob (float): 데이터 증강을 적용할 확률.
        """
        self.sequences = paired_sequences
        self.is_train = is_train
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        하나의 샘플을 반환합니다.
        학습(is_train=True) 시, 일정 확률로 데이터 증강(아이템 마스킹)을 적용합니다.
        """
        # (name_embed_list, event_id_list), target_id
        (name_embed_list, event_id_list), target_item = self.sequences[idx]
        
        # 데이터 증강 (학습 시에만, 그리고 시퀀스 길이가 2 이상일 때만 적용)
        if self.is_train and len(name_embed_list) > 1 and random.random() < self.augmentation_prob:
            # 마스킹할 아이템의 인덱스를 무작위로 선택 (마지막 아이템 제외)
            mask_idx = random.randint(0, len(name_embed_list) - 2)
            
            # 리스트에서 해당 인덱스의 아이템을 제거
            name_embed_list.pop(mask_idx)
            event_id_list.pop(mask_idx)

        name_embed_tensors = torch.stack([t for t in name_embed_list])
        event_id_tensors = torch.tensor(event_id_list)

        return (name_embed_tensors, event_id_tensors), target_item