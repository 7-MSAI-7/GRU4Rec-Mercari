import torch
import torch.nn as nn

class GruModel(nn.Module):
    def __init__(
        self,
        device,
        name_embedding_dim,
        event_embedding_dim,
        gru_hidden_dim,
        n_events,
        n_items,
        gru_num_layers=1,
        dropout_rate=0.3,
        **kwargs, # 사용되지 않는 추가 인자들을 흡수합니다.
    ):
        super(GruModel, self).__init__()
        
        self.device = device


        # 이벤트 임베딩 레이어
        self.event_embedding = nn.Embedding(
            num_embeddings=n_events, embedding_dim=event_embedding_dim, padding_idx=0
        )

        combined_embedding_dim = name_embedding_dim + event_embedding_dim

        # GRU 레이어
        self.gru = nn.GRU(
            input_size=combined_embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
        )

        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout_rate)

        # 최종 출력 레이어
        self.fc = nn.Linear(
            gru_hidden_dim, n_items
        )

    def forward(self, name_embeds, event_seq):

        # 이벤트
        event_embeds = self.event_embedding(event_seq)

        # 이름 임베딩, 아이템 임베딩, 이벤트 임베딩을 결합합니다.
        combined_embeds = torch.cat(
            (name_embeds, event_embeds), dim=2
        )

        # GRU 레이어를 통과시킵니다.
        output, _ = self.gru(combined_embeds)

        # GRU의 마지막 타임스텝의 출력에 드롭아웃을 적용합니다.
        output = self.dropout(output)

        # 시퀀스의 마지막 타임스텝의 출력을 사용합니다.
        output = output[:, -1, :]  

        # 최종 선형 레이어를 통과시켜 예측 점수를 얻습니다.
        logits = self.fc(
            output
        )

        return logits
