"""UVDoc で1枚を展開する。

上流 demo.py と同じ手順。違いは2点だけ:
  - 配布重みが CUDA 保存なので map_location="cpu" で読む
  - 展開は元解像度の画像に対して行う（出力解像度は入力と同じ）
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .model import UVDocnet
from .utils import IMG_SIZE, bilinear_unwarping

#: 上流同梱の学習済み重み（fork で uvdoc/model/ 配下へ移してある）。
DEFAULT_CKPT = Path(__file__).resolve().parent / "model" / "best_model.pkl"


def load_model(ckpt_path=DEFAULT_CKPT, device: str = "cpu"):
    model = UVDocnet(num_filter=32, kernel_size=5)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


def unwarp(img_bgr: np.ndarray, model=None, device: str = "cpu") -> np.ndarray:
    """BGR 画像を受け取り、展開した BGR 画像を返す（サイズは入力と同じ）。"""
    if model is None:
        model = load_model(device=device)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        points, _ = model(inp)
    size = img.shape[:2][::-1]
    out = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.unsqueeze(points[0], dim=0),
        img_size=tuple(size),
    )
    out = (out[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
