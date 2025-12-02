import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import *
from encoder import *
from model import *

def collate_fn_ser(batch: List[Dict]) -> Dict:
    """
    你以后如果想加上 SpecAugment / noise / padding 方式改变，
    改这里就行，训练代码不用动。
    """
    # 对齐到当前 batch 内最大的长度
    lengths = torch.stack([x["length"] for x in batch], dim=0)  # [B]
    max_len = int(lengths.max().item())

    waveforms = []
    labels = []
    for x in batch:
        w = x["waveform"]
        pad_len = max_len - w.size(0)
        if pad_len > 0:
            w = torch.nn.functional.pad(w, (0, pad_len))
        waveforms.append(w.unsqueeze(0))  # [1, T]
        labels.append(x["label"].unsqueeze(0))

    waveforms = torch.cat(waveforms, dim=0)  # [B, T]
    labels = torch.cat(labels, dim=0)        # [B]

    return {
        "waveforms": waveforms,
        "lengths": lengths,
        "labels": labels,
    }

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # print("[DEBUG] batch keys:", batch.keys())
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(waveforms, lengths)      # [B, C]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = waveforms.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    返回: (平均 loss, 准确率)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for batch in dataloader:
        waveforms = batch["waveforms"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(waveforms, lengths)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()

        bs = waveforms.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        total_correct += correct

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def train_loop(
    train_manifest: str,
    val_manifest: Optional[str] = None,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    hubert_name: str = "facebook/hubert-base-ls960",
    freeze_hubert: bool = True,
    num_workers: int = 2,
    save_dir: str = "checkpoints_ser",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 先用训练集构建 label 映射（确保 train/val 一致）
    tmp_ds = EmotionDataset(manifest_path = train_manifest)
    label2id = tmp_ds.label2id
    num_classes = len(label2id)
    print("Labels:", label2id)

    # 2) 构建真正的 Dataset / DataLoader
    train_ds = EmotionDataset(manifest_path = train_manifest, label2id = label2id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_ser,
    )

    if val_manifest is not None and os.path.exists(val_manifest):
        val_ds = EmotionDataset(manifest_path = val_manifest, label2id = label2id)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_ser,
        )
    else:
        val_loader = None

    # 3) 构建模型（后续你要换 encoder / temporal_model，只改这里）
    encoder = HubertFrameEncoder(
        hubert_name=hubert_name,
        freeze=freeze_hubert,
    )
    # temporal_model = MeanPoolingTemporalModel()
    temporal_model = CNNTemporalModel(encoder.hidden_size)
    model = EmotionClassifierModel(
        encoder=encoder,
        temporal_model=temporal_model,
        num_classes=num_classes,
        hidden_size=temporal_model.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] train_loss = {train_loss:.4f}")

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"[Epoch {epoch}] val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}")

            # 简单保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(save_dir, "best_model.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "label2id": label2id,
                        "hubert_name": hubert_name,
                    },
                    ckpt_path,
                )
                print(f"  -> New best model saved to {ckpt_path}")
        else:
            # 没有验证集就每个 epoch 都存
            if epoch % 10 == 0:
                ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "label2id": label2id,
                        "hubert_name": hubert_name,
                    },
                    ckpt_path,
                )
                print(f"  -> Model saved to {ckpt_path}")


if __name__ == "__main__":
    """
    你需要先准备 train_manifest.jsonl / val_manifest.jsonl，格式例如：

    {"audio_path": "data/wav/001.wav", "label": "happy"}
    {"audio_path": "data/wav/002.wav", "label": "angry"}
    ...
    """
    train_manifest = "/home/feipiao/Downloads/Emotional Speech Dataset (ESD)/Emotion Speech Dataset/train_5000.jsonl"
    val_manifest = "/home/feipiao/Downloads/Emotional Speech Dataset (ESD)/Emotion Speech Dataset/val_5000.jsonl"
    # 没有可以改成 None

    train_loop(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        batch_size=8,
        num_epochs=100,
        lr=1e-4,
        hubert_name="facebook/hubert-base-ls960",
        freeze_hubert=True,
        num_workers=4,
        save_dir="checkpoints_ser",
    )
