import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torchaudio

from encoder import HubertFrameEncoder
from model import CNNTemporalModel, EmotionClassifierModel


def load_checkpoint_model(ckpt_path: str, device: torch.device):
    """
    加载 ckpt，构建模型，返回 (model, id2label)
    需要和 train.py 里的构造方式保持一致。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    label2id: Dict[str, int] = ckpt["label2id"]
    hubert_name: str = ckpt.get("hubert_name", "facebook/hubert-base-ls960")
    num_classes = len(label2id)
    id2label = {i: lab for lab, i in label2id.items()}

    # 这些参数要和 train_loop 里保持一致
    FREEZE_HUBERT = True          # 你训练时 freeze_hubert=True

    # 1) 构建 encoder（HuBERT）
    encoder = HubertFrameEncoder(
        hubert_name=hubert_name,
        freeze=FREEZE_HUBERT,
    )

    # 2) 构建 temporal_model（CNN），这里假设你是 CNNTemporalModel(input_dim, hidden_dim=256, ...)
    temporal_model = CNNTemporalModel(
        input_dim=encoder.hidden_size,
    )

    # 3) 顶层分类模型
    model = EmotionClassifierModel(
        encoder=encoder,
        temporal_model=temporal_model,
        num_classes=num_classes,
        hidden_size=temporal_model.hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, id2label


def load_audio_to_tensor(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    和 EmotionDataset 里的逻辑保持一致：
    - torchaudio.load
    - 多通道 → mono
    - 重采样到 target_sr
    返回形状: [T]
    """
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # 转成 mono
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    return wav.squeeze(0)  # [T]


@torch.no_grad()
def predict_one(
    model: nn.Module,
    id2label: Dict[int, str],
    wav_path: str,
    device: torch.device,
    target_sr: int = 16000,
    topk: int = 3,
):
    """
    对单条 wav 做情感预测，返回 [(label, prob), ...]
    """
    wav = load_audio_to_tensor(wav_path, target_sr=target_sr)  # [T]
    length = torch.tensor([wav.size(0)], dtype=torch.long)     # [1]

    waveforms = wav.unsqueeze(0).to(device)  # [1, T]
    lengths = length.to(device)              # [1]

    logits = model(waveforms, lengths)       # [1, num_classes]
    probs = torch.softmax(logits, dim=-1)[0] # [num_classes]

    k = min(topk, probs.numel())
    top_probs, top_ids = torch.topk(probs, k)

    results: List = []
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        label = id2label[idx]
        results.append((label, p))

    return results


def main():
    parser = argparse.ArgumentParser(description="Emotion recognition inference")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的模型 checkpoint 路径")
    parser.add_argument("--wav", type=str, required=True, help="要预测的 wav 文件路径")
    parser.add_argument("--sr", type=int, default=16000, help="目标采样率（和训练时一致）")
    parser.add_argument("--topk", type=int, default=3, help="输出 Top-K 结果")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    print(f"[INFO] 加载模型: {args.ckpt}")
    model, id2label = load_checkpoint_model(args.ckpt, device=device)

    print(f"[INFO] 对 {args.wav} 进行情感预测...")
    results = predict_one(
        model=model,
        id2label=id2label,
        wav_path=args.wav,
        device=device,
        target_sr=args.sr,
        topk=args.topk,
    )

    print("=== 预测结果 ===")
    for i, (label, prob) in enumerate(results, start=1):
        print(f"Top {i}: {label} (prob={prob:.4f})")

    top1_label, top1_prob = results[0]
    print(f"\n预测情感：{top1_label}")
    print(f"置信度：{top1_prob*100:.2f}%")


if __name__ == "__main__":
    main()
