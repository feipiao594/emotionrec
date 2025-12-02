from typing import Tuple

import torch

from transformers import HubertModel

from encoder import BaseFrameEncoder

class HubertFrameEncoder(BaseFrameEncoder):
    """
    使用 HuggingFace 的 HuBERT 做帧级特征编码。
    你后面要换成别的预训练模型，只需要实现同样接口即可。
    """
    def __init__(
        self,
        hubert_name: str = "facebook/hubert-base-ls960",
        freeze: bool = True,
    ):
        super().__init__()
        self.hubert: HubertModel = HubertModel.from_pretrained(hubert_name)
        self.hidden_size = self.hubert.config.hidden_size

        if freeze:
            for p in self.hubert.parameters():
                p.requires_grad = False

    def forward(
        self,
        waveforms: torch.Tensor,  # [B, T]
        lengths: torch.Tensor,    # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 简单归一化（防止振幅差异太大）
        # 这里你也可以抽象成单独的 preprocessor 模块
        mean = waveforms.mean(dim=1, keepdim=True)
        std = waveforms.std(dim=1, keepdim=True).clamp(min=1e-5)
        normed = (waveforms - mean) / std

        B, T = normed.shape
        device = normed.device

        # attention_mask: [B, T]，1 表示有效帧
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        outputs = self.hubert(
            input_values=normed,
            attention_mask=mask,
        )
        features = outputs.last_hidden_state  # [B, T', D]  (T' ≤ T，有下采样)

        # 特征长度：这里简单用全部 T'，更精细可以根据下采样比计算
        T_prime = features.size(1)
        feat_lengths = torch.full(
            (B,),
            T_prime,
            dtype=torch.long,
            device=device,
        )

        return features, feat_lengths