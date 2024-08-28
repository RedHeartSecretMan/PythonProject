from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


mode = "image"
if mode == "image":
    checkpoint = "./sam2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "./sam2/configs/sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    image = Image.open("../datas/images/truck.jpg")
    image = np.array(image.convert("RGB"))
    input_coords = np.array([[500, 375]])
    input_labels = np.array([1])
    multimask_output = True

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_coords,
            point_labels=input_labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

elif mode == "video":
    checkpoint = "./sam2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "./sam2/configs/sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    video_dir = "../datas/videos/bedroom"
    inference_state = predictor.init_state(video_path=video_dir)
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
        frame_index, object_indexes, mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_index,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        # propagate the prompts to get masklets throughout the video
        video_segments = {}
        for frame_index, object_indexes, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[frame_index] = {
                object_index: (mask_logits[i] > 0.0).cpu().numpy()
                for i, object_index in enumerate(object_indexes)
            }
