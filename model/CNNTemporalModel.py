import torch
import torch.nn as nn

from model import BaseTemporalModel

class CNNTemporalModel(BaseTemporalModel):
    """
    使用 1D CNN 在时间维上做建模，然后通过 masked mean pooling
    得到一个句子级别的向量 [B, H]。

    输入: features [B, T, D]
    输出: embedding [B, hidden_dim]
    """
    def __init__(
        self,
        input_dim: int,          # D, 一般是 encoder.hidden_size (比如 HuBERT 的 hidden_size)
        hidden_dim: int = 256,   # CNN 通道数 / 输出维度
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
    
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim

        layers = []
        in_ch = input_dim

        for i in range(num_layers):
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # 保持时间长度不变
            )
            bn = nn.BatchNorm1d(hidden_dim)
            block = nn.Sequential(
                conv,
                bn,
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            layers.append(block)
            in_ch = hidden_dim

        self.cnn = nn.ModuleList(layers)

    def forward(
        self,
        features: torch.Tensor,   # [B, T, D]
        lengths: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """
        返回句子级 embedding: [B, hidden_dim]
        """
        B, T, D = features.shape
        device = features.device

        # 先转成 [B, D, T] 方便 Conv1d 在时间维卷积
        x = features.transpose(1, 2)  # [B, D, T]

        for block in self.cnn:
            residual = x
            x = block(x)             # [B, hidden_dim, T]
            # 残差连接要求通道数相同
            if self.use_residual and residual.shape == x.shape:
                x = x + residual

        # x: [B, hidden_dim, T]
        x = x.transpose(1, 2)        # [B, T, hidden_dim]

        # masked mean pooling
        # 构建 [B, T, 1] 的 mask
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
        mask = mask.unsqueeze(-1)  # [B, T, 1]

        x = x * mask               # [B, T, hidden_dim]
        sum_x = x.sum(dim=1)       # [B, hidden_dim]
        lengths = lengths.clamp(min=1).unsqueeze(1)  # [B, 1]
        mean_x = sum_x / lengths   # [B, hidden_dim]

        return mean_x
