import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from PIL import Image
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


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
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def draw_anns(anns, save_path, borders=True):
    if len(anns) == 0:
        return

    # Sort annotations by area
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)

    # Create a blank image with alpha channel
    img_height, img_width = sorted_anns[0]["segmentation"].shape
    img = np.ones((img_height, img_width, 4), dtype=np.float32)
    img[:, :, 3] = 0  # Alpha channel initialization to 0

    for ann in sorted_anns:
        m = ann["segmentation"]

        # Generate a random color with alpha value 0.5
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        color_mask_bgr = tuple(
            (color_mask[:3] * 255).astype(int)
        )  # Convert to BGR color

        # Apply color to the mask
        img[m] = np.concatenate(
            [
                np.full(m.shape, color_mask_bgr[0]),
                np.full(m.shape, color_mask_bgr[1]),
                np.full(m.shape, color_mask_bgr[2]),
                np.full(m.shape, color_mask[3]),
            ],
            axis=-1,
        )

        if borders:
            # Find contours
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Smooth contours and draw them
            for contour in contours:
                contour = cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                cv2.drawContours(
                    img, [contour], -1, (0, 0, 255, 0.4), thickness=1
                )

    img_uint8 = (img * 255).astype(np.uint8)
    cv2.imwrite(save_path, img_uint8)


mode = "image"
if mode == "image":
    sam2_checkpoint = "./stores/sam2/checkpoints/sam2_hiera_large.pt"
    sam2_config = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device.type)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    image = Image.open("./datas/images/truck.jpg")
    image = np.array(image.convert("RGB"))
    point_coords = np.array([[500, 375]])
    point_labels = np.array([1])
    multimask_output = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam2_predictor.set_image(image)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

elif mode == "video":
    sam2_checkpoint = "./stores/sam2/checkpoints/sam2_hiera_base_plus.pt"
    sam2_config = "sam2_hiera_b+.yaml"
    sam2_predictor = build_sam2_video_predictor(
        sam2_config, sam2_checkpoint, device=device.type
    )
    video_dir = "./datas/videos/bedroom"
    inference_state = sam2_predictor.init_state(video_path=video_dir)
    frame_index = 0
    obj_id = 1
    coords = np.array([[210, 350]], dtype=np.float32)
    labels = np.array([1], np.int32)
    multimask_output = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # add new prompts and instantly get the output on the same frame
        frame_index, object_indexes, mask_logits = sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_index,
            obj_id=obj_id,
            points=coords,
            labels=labels,
        )
        # propagate the prompts to get masklets throughout the video
        video_segments = {}
        for (
            frame_index,
            object_indexes,
            mask_logits,
        ) in sam2_predictor.propagate_in_video(inference_state):
            video_segments[frame_index] = {
                object_index: (mask_logits[i] > 0.0).cpu().numpy()
                for i, object_index in enumerate(object_indexes)
            }
elif mode == "automatic_mask":
    sam2_checkpoint = "../stores/sam2/checkpoints/sam2_hiera_small.pt"
    model_config = "sam2_hiera_s.yaml"
    sam2_model = build_sam2(
        model_config, sam2_checkpoint, device=device.type, apply_postprocessing=False
    )
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
    image = Image.open("./datas/images/truck.jpg")
    image = np.array(image.convert("RGB"))
    masks = mask_generator.generate(image)
