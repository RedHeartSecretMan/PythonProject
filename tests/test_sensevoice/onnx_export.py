import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "SenseVoice")
    )
)

import torch
from model import SenseVoiceSmall
from utils import export_utils

quantize = False

model_dir = "./stores/checkpoints/SenseVoiceSmall"
model, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="mps")

rebuilt_model = model.export(type="onnx", quantize=False)
model_path = kwargs.get("output_dir", os.path.dirname(kwargs.get("init_param")))

model_file = os.path.join(model_path, "model.onnx")
if quantize:
    model_file = os.path.join(model_path, "model_quant.onnx")

# export model
if not os.path.exists(model_file):
    with torch.no_grad():
        del kwargs["model"]
        export_dir = export_utils.export(model=rebuilt_model, **kwargs)
        print("Export model onnx to {}".format(model_file))
