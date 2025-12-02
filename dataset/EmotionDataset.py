import json
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
import torchaudio
########################################
#   数据集：EmotionDataset（JSONL 示例）
########################################

class EmotionDataset(Dataset):
    """
    JSONL 格式示例：
    {"audio_path": "data/wav/001.wav", "label": "happy"}
    {"audio_path": "data/wav/002.wav", "label": "angry"}
    ... 

    你以后要换成别的格式（比如 CSV / 文件夹结构），
    只需要写另一个 Dataset 类，实现 __len__ / __getitem__ 即可。
    """
    def __init__(
        self,
        manifest_path: str,
        label2id: Optional[Dict[str, int]] = None,
        target_sr: int = 16000,
    ):
        super().__init__()
        self.items = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)

        # 构建 / 复用 label 映射
        if label2id is None:
            labels = sorted({x["label"] for x in self.items})
            self.label2id = {lab: i for i, lab in enumerate(labels)}
        else:
            self.label2id = label2id

        self.id2label = {i: lab for lab, i in self.label2id.items()}

        self.target_sr = target_sr
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sr
            )
        return self._resamplers[orig_sr]

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [channels, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # 转 mono
        if sr != self.target_sr:
            resampler = self._get_resampler(sr)
            wav = resampler(wav)
        return wav.squeeze(0)  # [T]

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        wav = self._load_audio(item["audio_path"])
        label = item["label"]
        label_id = self.label2id[label]

        return {
            "waveform": wav,                 # [T]
            "length": torch.tensor(len(wav), dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }