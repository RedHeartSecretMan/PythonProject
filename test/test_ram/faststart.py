import os
import sys
import torch
from PIL import Image

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "recognize-anything",
        )
    )
)

from ram.models import ram_plus
from ram import get_transform, inference_ram


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. Model is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


if __name__ == "__main__":
    image_path = "datas/images/demo1.jpg"
    image_size = 384
    weight_path = "stores/checkpoints/ram_plus_swin_large_14m.pth"

    model = ram_plus(pretrained=weight_path, image_size=image_size, vit="swin_l")
    model.eval()
    model = model.to(device)
    image = get_transform(image_size=image_size)(Image.open(image_path))
    image = torch.as_tensor(image).unsqueeze(0).to(device)
    result = inference_ram(image, model)

    print("Image Tags: ", result[0])
    print("图像标签: ", result[1])
