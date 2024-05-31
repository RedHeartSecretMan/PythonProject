import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from RobustVideoMatting.inference_utils import VideoReader, VideoWriter
from RobustVideoMatting.model import MattingNetwork

model = MattingNetwork(variant="resnet50").eval()  # 或 variant="resnet50"
weights = torch.load(
    "/Users/WangHao/学习/PyCharm/stablediffusion/RobustVideoMatting/weights/rvm_resnet50.pth"
)
model.load_state_dict(weights)

reader = VideoReader(
    "/Users/WangHao/Desktop/屏幕录制2022-07-22 22.55.53.mov", transform=ToTensor()
)
writer = VideoWriter("output1.mp4", frame_rate=30)

bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1)  # 绿背景
rec = [None] * 4  # 初始记忆

with torch.no_grad():
    for src in DataLoader(reader):
        fgr, pha, *rec = model(
            src, *rec, downsample_ratio=0.25
        )  # 将上一帧的记忆给下一帧
        writer.write(fgr * pha + bgr * (1 - pha))

writer.close()
