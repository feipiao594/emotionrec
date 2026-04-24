import random
from typing import List, Optional

import torch
import torch.nn.functional as F


class AudioAugment:
    def __init__(
        self,
        noise_prob: float = 0.5,
        volume_prob: float = 0.3,
        speed_prob: float = 0.3,
        time_mask_prob: float = 0.2,
        snr_db_min: float = 10.0,
        snr_db_max: float = 20.0,
        volume_min: float = 0.7,
        volume_max: float = 1.3,
        speed_choices: Optional[List[float]] = None,
        time_mask_ratio_min: float = 0.05,
        time_mask_ratio_max: float = 0.10,
        max_transforms_per_sample: int = 2,
    ):
        self.noise_prob = noise_prob
        self.volume_prob = volume_prob
        self.speed_prob = speed_prob
        self.time_mask_prob = time_mask_prob
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.volume_min = volume_min
        self.volume_max = volume_max
        self.speed_choices = speed_choices or [0.9, 1.0, 1.1]
        self.time_mask_ratio_min = time_mask_ratio_min
        self.time_mask_ratio_max = time_mask_ratio_max
        self.max_transforms_per_sample = max(1, max_transforms_per_sample)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.numel() == 0:
            return wav

        aug_wav = wav.clone()
        transforms = self._sample_transforms()

        for name in transforms:
            if name == "volume":
                aug_wav = self._apply_volume(aug_wav)
            elif name == "speed":
                aug_wav = self._apply_speed(aug_wav)
            elif name == "noise":
                aug_wav = self._apply_white_noise(aug_wav)
            elif name == "time_mask":
                aug_wav = self._apply_time_mask(aug_wav)

        return aug_wav.clamp_(-1.0, 1.0)

    def _sample_transforms(self) -> List[str]:
        candidates: List[str] = []
        if random.random() < self.volume_prob:
            candidates.append("volume")
        if random.random() < self.speed_prob:
            candidates.append("speed")
        if random.random() < self.noise_prob:
            candidates.append("noise")
        if random.random() < self.time_mask_prob:
            candidates.append("time_mask")

        if len(candidates) <= self.max_transforms_per_sample:
            return candidates
        return random.sample(candidates, self.max_transforms_per_sample)

    def _apply_volume(self, wav: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(self.volume_min, self.volume_max)
        return wav * scale

    def _apply_speed(self, wav: torch.Tensor) -> torch.Tensor:
        speed = random.choice(self.speed_choices)
        if abs(speed - 1.0) < 1e-6:
            return wav

        target_len = max(1, int(round(wav.numel() / speed)))
        warped = F.interpolate(
            wav.view(1, 1, -1),
            size=target_len,
            mode="linear",
            align_corners=False,
        )
        return warped.view(-1)

    def _apply_white_noise(self, wav: torch.Tensor) -> torch.Tensor:
        signal_rms = wav.pow(2).mean().sqrt()
        if signal_rms.item() < 1e-8:
            return wav

        noise = torch.randn_like(wav)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-8)

        snr_db = random.uniform(self.snr_db_min, self.snr_db_max)
        noise_scale = signal_rms / (10 ** (snr_db / 20.0))
        scaled_noise = noise * (noise_scale / noise_rms)
        return wav + scaled_noise

    def _apply_time_mask(self, wav: torch.Tensor) -> torch.Tensor:
        num_samples = wav.numel()
        if num_samples < 100: # flag
            return wav

        ratio = random.uniform(self.time_mask_ratio_min, self.time_mask_ratio_max)
        mask_len = max(1, int(round(num_samples * ratio)))
        # mask ratio percent
        if mask_len >= num_samples / 2:
            mask_len = num_samples / 2 - 1
        start = random.randint(0, num_samples - mask_len)

        masked = wav.clone()
        masked[start:start + mask_len] = 0.0
        return masked
