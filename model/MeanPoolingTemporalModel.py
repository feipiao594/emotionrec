import torch
from model import BaseTemporalModel

class MeanPoolingTemporalModel(BaseTemporalModel):
    """
    最简单的时序池化：对时间维做 masked mean。
    以后你可以换成:
      - CNN
      - Transformer
      - BiLSTM
      - Attention pooling 等
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,    # [B, T', D]
        lengths: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        B, T, D = features.shape
        device = features.device

        # 构建 mask: [B, T]
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # [B, T, 1]

        # masked sum / length
        features = features * mask  # [B, T, D]
        sum_feat = features.sum(dim=1)  # [B, D]
        lengths = lengths.clamp(min=1).unsqueeze(1)  # [B, 1]
        mean_feat = sum_feat / lengths  # [B, D]

        return mean_feat