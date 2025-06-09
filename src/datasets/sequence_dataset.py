import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, paired_sequences):
        """
        초기화 메서드
        - paired_sequences: ((name_embeddings, event_ids), target_item_id) 형태의 튜플 리스트
        """
        self.sequences = paired_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        하나의 샘플을 반환합니다.
        DataLoader가 배치 처리를 할 수 있도록 모든 텐서를 CPU로 통일합니다.
        """
        # (name_embed_list, event_id_list), target_id
        input_seq, target_item = self.sequences[idx]
        
        # 입력 시퀀스의 텐서들을 CPU로 이동시켜 스택합니다.
        # 이 과정에서 발생하는 'device' 불일치 오류를 해결합니다.
        name_embed_list = torch.stack([t.to('cpu') for t in input_seq[0]])
        event_id_list = torch.tensor(input_seq[1])

        return (name_embed_list, event_id_list), target_item