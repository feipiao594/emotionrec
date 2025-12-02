import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtCore import Qt


class RecorderWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("简单录音器")
        self.resize(300, 200)

        # 录音参数
        self.sample_rate = 16000
        self.channels = 1

        # 状态
        self.stream = None
        self.audio_frames = []
        self.is_recording = False

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.btn = QPushButton("按住开始录音", self)
        self.btn.setCheckable(True)
        self.btn.setStyleSheet("font-size: 16px; padding: 20px;")
        layout.addWidget(self.btn, alignment=Qt.AlignCenter)

        # 按下/松开 录音
        self.btn.pressed.connect(self.start_recording)
        self.btn.released.connect(self.stop_recording)

    def audio_callback(self, indata, frames, time_, status):
        if status:
            print("Status:", status, flush=True)
        # indata: [frames, channels], float32
        # 这里直接复制保存
        self.audio_frames.append(indata.copy())

    def start_recording(self):
        if self.is_recording:
            return

        self.audio_frames = []
        self.is_recording = True
        self.btn.setText("录音中... 松开停止")
        self.btn.setStyleSheet("font-size: 16px; padding: 20px; background-color: red; color: white;")

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
            self.btn.setStyleSheet("font-size: 16px; padding: 20px;")
            QMessageBox.critical(self, "错误", f"无法启动录音设备：\n{e}")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.btn.setChecked(False)
        self.btn.setText("按住开始录音")
        self.btn.setStyleSheet("font-size: 16px; padding: 20px;")

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
        # 归一化到 int16
        audio_int16 = np.int16(audio * 32767)

        # 文件名：带时间戳
        filename = time.strftime("record_%Y%m%d_%H%M%S.wav")
        write(filename, self.sample_rate, audio_int16)
        print(f"录音已保存到：{filename}")
        QMessageBox.information(self, "完成", f"录音已保存到：\n{filename}")


def main():
    app = QApplication(sys.argv)
    w = RecorderWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
