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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = MattingNetwork(variant="resnet50").eval().to(device)
weights = torch.load("stores/weights/rvm_resnet50.pth")
model.load_state_dict(weights)

input_dir = "datas/mp4"
input_path = os.path.join(input_dir, "1.mp4")
reader = VideoReader(input_path, transform=ToTensor())
output_dir = "results/mp4"
output_path = os.path.join(output_dir, "1.mp4")
os.makedirs(output_dir, exist_ok=True)
writer = VideoWriter(output_path, frame_rate=reader.frame_rate)

bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1).to(device)  # 绿背景
rec = [None] * 4  # 初始状态
downsample_ratio = 0.125  # 下采样率1080P使用0.25，4K使用0.125

with torch.no_grad():
    for src in DataLoader(reader):
        fgr, pha, *rec = model(src.to(device), *rec, downsample_ratio)
        com = fgr * pha + bgr * (1 - pha)
        writer.write(com)
