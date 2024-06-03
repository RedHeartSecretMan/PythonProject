import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "RobustVideoMatting"
        )
    )
)
from inference_utils import VideoReader, VideoWriter
from model import MattingNetwork

model = MattingNetwork(variant="resnet50").eval()  # 或 variant="resnet50"
weights = torch.load("robustvideomatting/weights/rvm_resnet50.pth")
model.load_state_dict(weights)

input_dir = "datas/mp4"
input_path = os.path.join(input_dir, "1.mp4")
reader = VideoReader(input_path, transform=ToTensor())
output_dir = "results/mp4"
output_path = os.path.join(output_dir, "1.mp4")
os.makedirs(output_dir, exist_ok=True)
writer = VideoWriter(output_path, frame_rate=30)

bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1)  # 绿背景
rec = [None] * 4  # 初始状态
downsample_ratio = 0.25  # 下采样率

with torch.no_grad():
    for src in DataLoader(reader):
        fgr, pha, *rec = model(src, *rec, downsample_ratio)
        com = fgr * pha + bgr * (1 - pha)
        writer.write(com)
