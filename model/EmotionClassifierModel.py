########################################
#   顶层模型：EmotionClassifierModel
########################################

import torch
import torch.nn as nn
from typing import Optional

import model, encoder

class EmotionClassifierModel(nn.Module):
    """
    顶层情感识别模型：
      encoder: BaseFrameEncoder (HuBERT / Wav2Vec2 / 你自己的)
      temporal_model: BaseTemporalModel (MeanPooling / CNN / Transformer 等)
      classifier: 线性分类头

    你以后替换任何一块，只要保持接口一致，这个类不用改。
    """
    def __init__(
        self,
        encoder: encoder.BaseFrameEncoder,
        temporal_model: model.BaseTemporalModel,
        num_classes: int,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_model = temporal_model

        # 默认从 encoder 中读 hidden_size（比如 HuBERT 的 hidden_size）
        if hidden_size is None:
            if hasattr(encoder, "hidden_size"):
                hidden_size = encoder.hidden_size
            else:
                raise ValueError(
                    "hidden_size 必须显式传入，或者 encoder 需要有 .hidden_size 属性"
                )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        waveforms: torch.Tensor,  # [B, T]
        lengths: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        # 1) 帧级编码
        features, feat_lengths = self.encoder(waveforms, lengths)  # [B, T', D], [B]

        # 2) 时序池化 / 建模
        utter_emb = self.temporal_model(features, feat_lengths)    # [B, H]

        # 3) 分类
        logits = self.classifier(utter_emb)                        # [B, C]
        return logits
