from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

import logging
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.utils import instantiate


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    # Assuming config_file is a path to the configuration file, extract the directory and file name
    config_dir = "/".join(config_file.split("/")[:-1])  # Directory path
    config_name = config_file.split("/")[-1].replace(
        ".yaml", ""
    )  # File name without extension

    # Initialize Hydra with the config path
    with initialize(config_path=config_dir):
        cfg = compose(config_name=config_name, overrides=hydra_overrides_extra)

    # Resolve any interpolations in the configuration
    OmegaConf.resolve(cfg)

    # Instantiate the model from the configuration
    model = instantiate(cfg.model, _recursive_=True)

    # Load the checkpoint if provided
    if ckpt_path:
        _load_checkpoint(model, ckpt_path)

    # Move the model to the specified device
    model = model.to(device)

    # Set the model to evaluation mode if specified
    if mode == "eval":
        model.eval()

    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


mode = "image"
if mode == "image":
    sam2_checkpoint = "./sam2/checkpoints/sam2_hiera_large.pt"
    sam2_config = "./sam2/configs/sam2_hiera_l.yaml"
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    image = Image.open("./datas/images/truck.jpg")
    image = np.array(image.convert("RGB"))
    input_coords = np.array([[500, 375]])
    input_labels = np.array([1])
    multimask_output = True

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam2_predictor.set_image(image)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=input_coords,
            point_labels=input_labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

elif mode == "video":
    sam2_checkpoint = "./sam2/checkpoints/sam2_hiera_large.pt"
    sam2_config = "./sam2/configs/sam2_hiera_l.yaml"
    sam2_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)
    video_dir = "../datas/videos/bedroom"
    inference_state = sam2_predictor.init_state(video_path=video_dir)
    frame_index = 0
    obj_id = 1
    # Let's add a positive click at (x, y) = (210, 350) to get started
    # for labels, `1` means positive click and `0` means negative click
    points = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)
    input_coords = np.array([[500, 375]])
    input_labels = np.array([1])
    multimask_output = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # add new prompts and instantly get the output on the same frame
        frame_index, object_indexes, mask_logits = sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_index,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        # propagate the prompts to get masklets throughout the video
        video_segments = {}
        for frame_index, object_indexes, mask_logits in sam2_predictor.propagate_in_video(
            inference_state
        ):
            video_segments[frame_index] = {
                object_index: (mask_logits[i] > 0.0).cpu().numpy()
                for i, object_index in enumerate(object_indexes)
            }
