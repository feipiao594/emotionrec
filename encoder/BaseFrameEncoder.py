from typing import Tuple
import torch
import torch.nn as nn

class BaseFrameEncoder(nn.Module):
    """
    帧级特征编码器接口：
    输入: 波形 [B, T], 长度 [B]
    输出: 特征 [B, T', D], 特征长度 [B]
    """
    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError