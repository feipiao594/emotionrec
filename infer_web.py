# server.py
import io
import os
import shutil
from typing import Dict, Optional

import time
import torch
import torch.nn as nn
import torchaudio
import subprocess
import tempfile

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    Depends,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

from encoder import HubertFrameEncoder
from model import CNNTemporalModel, EmotionClassifierModel

# ==== 配置部分 ====
CKPT_PATH = "checkpoints_ser/best_loss_model_new_35000.pt"
TARGET_SR = 16000
SECRET_KEY = "secret_key"  # 生产环境请改！！！
USERNAME = "feipiao"  # demo用，生产请接数据库或别的认证
PASSWORD = "password" # demo用，生产请加密存储
# ==================

app = FastAPI()

# 会话中间件：用于 Cookie Session（保存登录状态）
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# 如果以后有跨域需求（比如前后端分离），可以打开 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段随便，生产请收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="web")


class PredictResponse(BaseModel):
    top1_label: str
    top1_prob: float
    topk: Dict[str, float]


def load_model(ckpt_path: str, device: torch.device):
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


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model, id2label = load_model(CKPT_PATH, device=device)

def load_audio_from_bytes(data: bytes, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    从浏览器上传的 webm/ogg 等音频字节中读取波形。
    步骤：
    1. 把 bytes 写到临时 .webm 文件
    2. 用 ffmpeg 转成 16kHz 单声道 wav
    3. 用 torchaudio.load 读进来
    """
    # 1) 写入临时输入文件
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f_out:

        f_in.write(data)
        f_in.flush()

        # 2) 调用 ffmpeg 转码到 wav (16k, mono)
        cmd = [
            "ffmpeg",
            "-y",             # 覆盖输出
            "-i", f_in.name,  # 输入文件
            "-ac", "1",       # 单声道
            "-ar", str(target_sr),  # 采样率
            f_out.name,       # 输出 wav
        ]
        # 静音运行 ffmpeg
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        try:
            out_size = os.path.getsize(f_out.name)
            if out_size < 500 * 1024:  # 500KB
                os.makedirs("temp", exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_name = f"{ts}.wav"
                save_path = os.path.join("temp", save_name)
                shutil.copy2(f_out.name, save_path)
                print(f"[INFO] Saved converted wav ({out_size} bytes) to {save_path}")
        except Exception as e:
            # 保存失败不影响后续推理，打个日志就行
            print(f"[WARN] Failed to save small wav file: {e}")

        # 3) 用 torchaudio 读转换后的 wav
        wav, sr = torchaudio.load(f_out.name)  # [C, T]

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 理论上已经是 target_sr 了，这里再保险一次
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)

    return wav.squeeze(0)  # [T]



# ============ 登录相关 ============

def get_current_user(request: Request) -> Optional[str]:
    """
    从 session 中取当前用户名
    """
    return request.session.get("user")


def require_login(request: Request) -> str:
    """
    依赖：需要登录，否则跳转到 /login
    """
    user = get_current_user(request)
    if not user:
        # 303 跳转到登录页
        return RedirectResponse(url="/login", status_code=303)
    return user


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    根路径：如果已登录 -> 跳到 /app
            未登录 -> 跳到 /login
    """
    if get_current_user(request):
        return RedirectResponse(url="/app", status_code=303)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    登录页（GET）：显示登录表单
    """
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """
    登录表单提交（POST），简单用户名密码校验
    """
    if username == USERNAME and password == PASSWORD:
        request.session["user"] = username
        return RedirectResponse(url="/app", status_code=303)
    # 登录失败，回到登录页并带上错误提示
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": "用户名或密码错误",
        },
        status_code=401,
    )


@app.get("/logout")
async def logout(request: Request):
    """
    登出：清空 session
    """
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    """
    录音 & 上传 前端页面
    需要登录
    """
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        "app.html",
        {
            "request": request,
            "username": user,
        },
    )


# ============ 预测接口 ============

@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
async def predict(request: Request, file: UploadFile = File(...)):
    # 也可以在这里检查是否已登录：
    if not get_current_user(request):
        # 未登录就不让用预测接口
        return RedirectResponse(url="/login", status_code=303)

    audio_bytes = await file.read()

    wav = load_audio_from_bytes(audio_bytes, target_sr=TARGET_SR).to(device)  # [T]

    length = torch.tensor([wav.size(0)], dtype=torch.long, device=device)
    waveforms = wav.unsqueeze(0)  # [1, T]

    logits = model(waveforms, length)
    probs = torch.softmax(logits, dim=-1)[0]

    topk = min(3, probs.numel())
    top_probs, top_ids = torch.topk(probs, topk)

    topk_dict: Dict[str, float] = {}
    for p, idx in zip(top_probs.tolist(), top_ids.tolist()):
        label = id2label[idx]
        topk_dict[label] = p

    top1_label = id2label[int(top_ids[0])]
    top1_prob = float(top_probs[0])

    return PredictResponse(
        top1_label=top1_label,
        top1_prob=top1_prob,
        topk=topk_dict,
    )
