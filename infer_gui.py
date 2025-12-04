import sys
import os
import time
from typing import Dict, List

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

import torch
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QLabel,
    QCheckBox,
)
from PyQt5.QtCore import Qt

from encoder import HubertFrameEncoder
from model import CNNTemporalModel, EmotionClassifierModel

CKPT_PATH = "checkpoints_ser/best_model.pt"
TARGET_SR = 16000
CHANNELS = 1

def load_checkpoint_model(ckpt_path: str, device: torch.device):
    """
    加载 ckpt，构建模型，返回 (model, id2label)
    必须和 train.py 里的构建逻辑保持一致。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    label2id: Dict[str, int] = ckpt["label2id"]
    hubert_name: str = ckpt.get("hubert_name", "facebook/hubert-base-ls960")
    num_classes = len(label2id)
    id2label = {i: lab for lab, i in label2id.items()}

    encoder = HubertFrameEncoder(
        hubert_name=hubert_name,
        freeze=True,
    )

    temporal_model = CNNTemporalModel(encoder.hidden_size)

    model = EmotionClassifierModel(
        encoder=encoder,
        temporal_model=temporal_model,
        num_classes=num_classes,
        hidden_size=temporal_model.hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, id2label


@torch.no_grad()
def predict_from_audio_array(
    model: nn.Module,
    id2label: Dict[int, str],
    audio: np.ndarray,   # shape: [N] or [N, 1], float32
    sample_rate: int,
    device: torch.device,
) -> List:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    wav = torch.from_numpy(audio).to(device)          # [T]
    length = torch.tensor([wav.size(0)], dtype=torch.long, device=device)
    waveforms = wav.unsqueeze(0)                      # [1, T]

    logits = model(waveforms, length)                 # [1, num_classes]
    probs = torch.softmax(logits, dim=-1)[0]          # [num_classes]

    k = min(3, probs.numel())
    top_probs, top_ids = torch.topk(probs, k)

    results = []
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        label = id2label[idx]
        results.append((label, p))

    return results


class RecorderWindow(QMainWindow):
    def __init__(self, model: nn.Module, id2label: Dict[int, str], device: torch.device):
        super().__init__()

        self.setWindowTitle("情感录音识别")
        self.resize(360, 240)

        self.model = model
        self.id2label = id2label
        self.device = device

        # 录音参数
        self.sample_rate = TARGET_SR
        self.channels = CHANNELS

        # 状态
        self.stream = None
        self.audio_frames = []
        self.is_recording = False

        # === UI ===
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.info_label = QLabel("按住下方按钮开始录音，松开结束并识别情感。")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # 新增：是否保存录音的按钮（复选框）
        self.save_checkbox = QCheckBox("保存录音为 WAV 文件")
        self.save_checkbox.setChecked(False)  # 默认不保存
        layout.addWidget(self.save_checkbox, alignment=Qt.AlignCenter)

        self.btn = QPushButton("按住开始录音", self)
        self.btn.setCheckable(True)
        self.btn.setStyleSheet("font-size: 18px; padding: 20px;")
        layout.addWidget(self.btn, alignment=Qt.AlignCenter)

        # 按下/松开 录音
        self.btn.pressed.connect(self.start_recording)
        self.btn.released.connect(self.stop_recording)

    def audio_callback(self, indata, frames, time_, status):
        if status:
            print("Status:", status, flush=True)
        self.audio_frames.append(indata.copy())

    def start_recording(self):
        if self.is_recording:
            return

        self.audio_frames = []
        self.is_recording = True
        self.btn.setText("录音中... 松开停止")
        self.btn.setStyleSheet("font-size: 18px; padding: 20px; background-color: red; color: white;")

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
            print("开始录音...")
        except Exception as e:
            self.is_recording = False
            self.btn.setChecked(False)
            self.btn.setText("按住开始录音")
            self.btn.setStyleSheet("font-size: 18px; padding: 20px;")
            QMessageBox.critical(self, "错误", f"无法启动录音设备：\n{e}")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.btn.setChecked(False)
        self.btn.setText("按住开始录音")
        self.btn.setStyleSheet("font-size: 18px; padding: 20px;")

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print("停止录音失败：", e)

        if not self.audio_frames:
            QMessageBox.information(self, "提示", "没有录到任何音频。")
            return

        audio = np.concatenate(self.audio_frames, axis=0)  # [N, channels]
        print(f"录音长度: {audio.shape[0]/self.sample_rate:.2f} 秒")

        filename = None
        # ✅ 只有勾选了“保存录音为 WAV 文件”时，才写文件
        if self.save_checkbox.isChecked():
            filename = time.strftime("record_%Y%m%d_%H%M%S.wav")
            audio_int16 = np.int16(audio * 32767)
            write(filename, self.sample_rate, audio_int16)
            print(f"录音已保存到：{filename}")
        else:
            print("未保存录音文件（用户未勾选保存选项）。")

        # 模型预测
        try:
            results = predict_from_audio_array(
                model=self.model,
                id2label=self.id2label,
                audio=audio,
                sample_rate=self.sample_rate,
                device=self.device,
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测时发生错误：\n{e}")
            return

        if not results:
            QMessageBox.warning(self, "警告", "没有得到任何预测结果。")
            return

        top1_label, top1_prob = results[0]

        msg_lines = [
            f"预测情感：{top1_label}",
            f"置信度：{top1_prob*100:.2f}%",
        ]
        if len(results) > 1:
            msg_lines.append("")
            msg_lines.append("其它可能性：")
            for label, prob in results[1:]:
                msg_lines.append(f"  - {label:<10} {prob*100:6.2f}%")

        if filename is not None:
            msg_lines.append("")
            msg_lines.append(f"录音已保存到：{filename}")
        else:
            msg_lines.append("")
            msg_lines.append("录音未保存（可勾选选项启用保存）。")

        msg = "\n".join(msg_lines)
        QMessageBox.information(self, "预测结果", msg)


def main():
    app = QApplication(sys.argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        print(f"加载模型: {CKPT_PATH}")
        model, id2label = load_checkpoint_model(CKPT_PATH, device=device)
        print("模型加载完成")
    except Exception as e:
        err_app = QMessageBox()
        err_app.setIcon(QMessageBox.Critical)
        err_app.setWindowTitle("错误")
        err_app.setText(f"加载模型失败：\n{e}")
        err_app.show()
        app.exec_()
        sys.exit(1)

    w = RecorderWindow(model=model, id2label=id2label, device=device)
    w.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
