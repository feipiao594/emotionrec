import torch
import torch.nn as nn

class BaseTemporalModel(nn.Module):
    """
    时序建模 / 池化接口：
    输入: 帧特征 [B, T', D], 帧长度 [B]
    输出: 句子级 embedding [B, H]
    """
    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError