import os
import re
import inspect
import argparse
from tqdm.auto import tqdm
from typing import List, Optional, Union

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from PIL import Image
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plot


def optional_str(string):
    return None if string == "None" else string


def image_grid(inputs, x=1, y=1):
    image_list = inputs
    assert len(image_list) == x*y
    width, height = image_list[0].size
    grid = Image.new("RGB", size=(x*width, y*height))
    k = 1
    for idx in range(x*y):
        if x <= y:
            if idx == k*y:
                k += 1
            grid.paste(image_list[idx], box=((idx-(k-1)*x)%y*width, idx//x*height))
        if x > y:
            grid.paste(image_list[idx], box=(idx%x*width, idx//x*height))
    outputs = grid
    
    return outputs


def show_image(image_list, prompt=0, scale=5, dpi=300, colormap=None):
    sizes = np.ceil(np.sqrt(np.array(len(image_list))))
    plot.figure(num=prompt, figsize=(sizes*scale, sizes*scale), dpi=dpi)
    for idx, image in enumerate(image_list):
        plot.subplot(int(sizes), int(sizes), idx+1)
        plot.imshow(image, cmap=colormap)
        plot.axis("off")
    plot.show() 
    

def save_image(image_list, save_path, prompt):
    regex = r"^[^/\\:\*\?\"\'\<\>\|]{1,255}"
    prompt = re.search(regex, prompt).group()
    if not os.path.exists(os.path.join(save_path, prompt)):
        os.makedirs(os.path.join(save_path, prompt))
    for image in image_list:
        fn_list = list(map(lambda string: os.path.splitext(string)[0], os.listdir(os.path.join(save_path, prompt))))
        k = 0
        for idx in range(len(fn_list)):
            try:
                fn_list[idx] = int(fn_list[idx])
            except:
                fn_list.remove(fn_list[idx-k])
                k += 1
        if len(fn_list) == 0:
            image.save(os.path.join(save_path, prompt, f"{str(0).zfill(4)}.png"))
        else:
            name_index = (set(fn_list) ^ set(range(max(fn_list) + 1))).pop() if len(set(fn_list)) != max(
                fn_list) + 1 else max(fn_list) + 1
            image.save(os.path.join(save_path, prompt, f"{str(name_index).zfill(4)}.png"))


# ???????????????????????????????????????
class StableDiffusionImgToImgPipeline(DiffusionPipeline):
    def __init__(self, vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: Union[DDIMScheduler,
                                  DDPMScheduler,
                                  PNDMScheduler,
                                  LMSDiscreteScheduler,
                                  EulerDiscreteScheduler,],
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPFeatureExtractor):
        super().__init__()
        self.register_modules(vae=vae,
                              text_encoder=text_encoder,
                              tokenizer=tokenizer,
                              unet=unet,
                              scheduler=scheduler,
                              safety_checker=safety_checker,
                              feature_extractor=feature_extractor)
    @staticmethod
    def preprocess(inputs):
        image = inputs
        width, height = image.size
        width, height = map(lambda x: x - x % 8, (width, height))
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        outputs = 2. * image - 1.

        return outputs

    @torch.no_grad()
    def __call__(self, prompt: Union[str, List[str]],
                       image: torch.FloatTensor,
                       strength: float = 0.75,
                       num_inference_steps: Optional[int] = 50,
                       guidance_scale: Optional[float] = 7.5,
                       eta: Optional[float] = 0.0,
                       generator: Optional[torch.Generator] = None,
                       output_type: Optional[str] = "pil"):

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"### \"prompt\" has to be of type str or list but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"### The value of strength should in (0.0, 1.0) but is {strength}")

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # encode the init image into latents and scale the latents
        init_latents = self.vae.encode(self.preprocess(image).to(self.device)).latent_dist.sample()
        init_latents = 0.18215 * init_latents

        # prepare init_latents noise to latents
        init_latents = torch.cat([init_latents] * batch_size)

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # get prompt text embeddings
        text_input = self.tokenizer(prompt,
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here guidance_scale is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (??) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to ?? in DDIM paper: https://arxiv.org/abs/2010.02502 and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run NSFW safety checker
        safety_cheker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
        image, has_nsfw_concept = self.safety_checker(images=image, clip_input=safety_cheker_input.pixel_values)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"images": image, "nsfw_content_detected": has_nsfw_concept}


# ???????????????????????????????????????
class TextToImageSetting:
    text2image = "0.0.1"
    model_id = "CompVis/stable-diffusion-v1-4"
    
    prompt_dict = {
        "0000": "a photo of an astronaut riding a horse on mars",
        "0001": "digital art of portrait of a woman, holding pencil, inspired, head-and-shoulders shot, white background, cute pixar character",
        "0002": "digital art of a man looking upwards, eyes wide inwonder, awestruck, in the style of Pixar, Up, character, white background",
        "0003": "the starry sky painting",
        "0004": "donald Trump wears a panda headgear",
        "0005": "a painting of a fox sitting in a field at sunrise in the style of Claude Monet",
        "0006": "dreams flowers and maidens",
        "0007": "teddy bears, working on new AI research, on the moon in the 1980s",
        "0008": "an astronaut, lounging in a tropical resort in space, as pixel art",
        "0009": "the whale was flying in the air, and below was a volcano and a snow-capped mountain",
        "0010": "robot, looking at the clouds hanging in the distance, solemn expression, strange background",
    }
    prompt = prompt_dict["0010"]

    seed = 3939590921
    sd_mode = "PNDM"
    
    num_inference_steps = 50
    guidance_scale = 7

    sample_num = 5
    batch = 1
    height = 512 
    width = 512
 
    device = "cpu"
    fp_mode = "fp32"
    save_dir = "./results"


# ?????????????????????????????????????????????
class ImageToImageSetting:
    image2image = "0.0.1"   
    model_id = "v1-4" 
      
    prompt_dict = {
        "0001": "a fantasy landscape, trending on artstation",
        }
    prompt = prompt_dict["0001"]
    image_url = "https://img-blog.csdnimg.cn/11fc2a2b3aa1471b9bced71ee0ddf507.jpeg"

    seed = 3939590921
    sd_mode = "DDIM" 
    
    num_inference_steps = 50
    guidance_scale = 7.5 
    strength = 0.75
      
    sample_num = 5
    batch = 1
    height = 512
    width = 512 
    
    device = "cpu"
    fp_mode = "fp32"
    save_dir = "./results"


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_task", default="text2image", type=str, choices=["text2image", "image2image"],
                        help="????????????")
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4", type=str, choices=["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"],
                        help="????????????")
    
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars", type=str,
                        help="????????? - ???????????????????????????????????????????????????, ????????????????????????????????????(horse, dog, human, mountain)?????????????????????(Portrait, Edgar, Rembrandt)???????????????(4k, beautiful, epic)")
    parser.add_argument("--image_url", default=None, type=optional_str,
                        help="?????????????????? - ????????????????????????????????????????????????")
    
    parser.add_argument("--seed", default=51, type=int,
                        help="???????????? - ?????????????????????????????????, ???????????????????????????, ??????????????????????????????????????????????????????????????????, ???????????????????????????????????????????????????????????????, ????????????????????????")
    parser.add_argument("--sd_mode", default=None, type=optional_str,
                        help="????????? - ??????????????????????????????????????????????????????????????????, ????????????????????????????????????????????????????????????????????????????????????")

    parser.add_argument("--num_inference_steps", default=50, type=int,
                        help="?????????????????? - ?????????????????????????????????????????????")
    parser.add_argument("--guidance_scale", default=7, type=int,
                        help="????????????????????? - ????????????????????????????????????, ????????????, ???????????? 0~20, ???????????????????????????????????????, ????????? 7~8.5")
    parser.add_argument("--strength", default=0.75, type=int,
                        help="???????????? - ???????????? 0~1, ?????????????????????????????????????????????")

    parser.add_argument("--sample_num", default=50, type=int,
                        help="??????????????????????????????????????????????????????????????????")
    parser.add_argument("--batch", default=1, type=int,
                        help="??????????????????????????? - ?????????????????????????????????????????????, 2 ?????????????????????????????? 2 ?????????, ???????????????????????? 1 ??????")
    parser.add_argument("--height", default=512, type=int,
                        help="????????????????????? - ????????? 8 ?????????(?????? 512 ?????????????????????, ?????????????????? 512 ????????????????????????)")
    parser.add_argument("--width", default=512, type=int,
                        help="????????????????????? - ????????? 8 ?????????(?????? 512 ?????????????????????, ?????????????????? 512 ????????????????????????)")


    parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"],
                        help="?????????????????????????????????????????????cuda??????")
    parser.add_argument("--fp_mode", default="fp16", type=str, choices=["fp32", "fp16"],
                        help="??????????????????????????????????????????")
    
    parser.add_argument("--save_dir", default="./results", type=str,
                        help="?????????????????? - ???????????? ./path ?????????????????????????????????????????? path ?????????, ????????????????????????")

    return parser.parse_args()


def get_pipe(settings):
    # ??????UNet???????????????
    if settings.sd_mode == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(settings.model_id,
                                                  subfolder="scheduler",
                                                  use_auth_token=True)
    elif settings.sd_mode == "DDPM":
        scheduler = DDPMScheduler.from_pretrained(settings.model_id,
                                                  subfolder="scheduler",
                                                  use_auth_token=True)
    elif settings.sd_mode == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(settings.model_id,
                                                  subfolder="scheduler",
                                                  use_auth_token=True)
    elif settings.sd_mode == "LMSD":
        scheduler = LMSDiscreteScheduler.from_pretrained(settings.model_id,
                                                         subfolder="scheduler",
                                                         use_auth_token=True)
    elif settings.sd_mode == "Euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(settings.model_id,
                                                           subfolder="scheduler",
                                                           use_auth_token=True)
    else:
        scheduler = None
    
    # ??????Diffusion??????
    if hasattr(settings, "text2image"):    
        if scheduler is not None:
            if settings.fp_mode == "fp32":
                pipe = StableDiffusionPipeline.from_pretrained(settings.model_id,
                                                               cheduler=scheduler,
                                                               torch_dtype=torch.float32,
                                                               use_auth_token=True)      
            elif settings.fp_mode == "fp16":
                pipe = StableDiffusionPipeline.from_pretrained(settings.model_id,
                                                               revision="fp16",
                                                               scheduler=scheduler,
                                                               torch_dtype=torch.float16,
                                                               use_auth_token=True)
            else:
                pass
        else:
            if settings.fp_mode == "fp32":
                pipe = StableDiffusionPipeline.from_pretrained(settings.model_id,
                                                               torch_dtype=torch.float32,
                                                               use_auth_token=True)      
            elif settings.fp_mode == "fp16":
                pipe = StableDiffusionPipeline.from_pretrained(settings.model_id,
                                                               revision="fp16",
                                                               torch_dtype=torch.float16,
                                                               use_auth_token=True)
            else:
                pass
    elif hasattr(settings, "image2image"):
        if scheduler is not None:
            if settings.fp_mode == "fp32":
                pipe = StableDiffusionImgToImgPipeline.from_pretrained(settings.model_id,
                                                                       scheduler=scheduler,
                                                                       use_auth_token=True)
            elif settings.fp_mode == "fp16":
                pipe = StableDiffusionImgToImgPipeline.from_pretrained(settings.model_id,
                                                                       revision="fp16",
                                                                       scheduler=scheduler,
                                                                       torch_dtype=torch.float16,
                                                                       use_auth_token=True)
            else:
                pass
        else:
            if settings.fp_mode == "fp32":
                pipe = StableDiffusionImgToImgPipeline.from_pretrained(settings.model_id,                                                            
                                                                       use_auth_token=True)
            elif settings.fp_mode == "fp16":
                pipe = StableDiffusionImgToImgPipeline.from_pretrained(settings.model_id,
                                                                       revision="fp16",
                                                                       torch_dtype=torch.float16,
                                                                       use_auth_token=True)
            else:
                pass           
    else:
        pass    
     
    return pipe.to(settings.device)


def main(_args):
    # ??????????????????
    if _args.model_task == "text2image":
        settings = TextToImageSetting()
    elif _args.model_task == "image2image":
        settings = ImageToImageSetting()      
    else:
        pass
    settings.model_id = _args.model_id  
    
    settings.prompt = _args.prompt
    if hasattr(settings, "image2image"):
        settings.image_url = _args.image_url
    
    settings.seed = _args.seed
    settings.sd_mode = _args.sd_mode
    
    settings.num_inference_steps = _args.num_inference_steps
    settings.guidance_scale = _args.guidance_scale
    if hasattr(settings, "image2image"):
        settings.strength = _args.strength

    settings.sample_num = _args.sample_num
    settings.batch = _args.batch
    settings.height = _args.height 
    settings.width = _args.width
 
    settings.device = _args.device
    settings.fp_mode = _args.fp_mode
    if settings.device == "cuda":
        if not torch.cuda.is_available():
            settings.device = "cpu"
        if settings.fp_mode == "fp16":
            settings.fp_mode = "fp32"
    if settings.device == "cpu":
        if settings.fp_mode == "fp16":
            settings.fp_mode = "fp32"
    settings.save_dir = _args.save_dir
    
    # ????????????????????????
    pipe = get_pipe(settings=settings)
    
    # ?????????????????????????????????
    image_list = []
    generator = torch.Generator(device=settings.device).manual_seed(settings.seed)
    if hasattr(settings, "text2image"):
        for idx in range(settings.sample_num):
            print(f"???????????????{idx+1}????????? - ??????{settings.batch}???")
            if settings.device == "cuda":
                with autocast("cuda"):
                    data = pipe([settings.prompt]*settings.batch,
                                height=settings.height, width=settings.width,
                                num_inference_steps=settings.num_inference_steps,
                                guidance_scale=settings.guidance_scale,
                                generator=generator)
            else:
                data = pipe([settings.prompt]*settings.batch,
                            height=settings.height, width=settings.width,
                            num_inference_steps=settings.num_inference_steps,
                            guidance_scale=settings.guidance_scale,
                            generator=generator)
            image = data["images"]
            save_image(image, settings.save_dir, settings.prompt)
            image_list.append(image_grid(image, x=settings.batch, y=settings.batch))
        show_image(image_list, settings.prompt)
    elif hasattr(settings, "image2image"):
        init_image = []
        try:
            init_image = Image.open(BytesIO(requests.get(settings.image_url).content)).convert("RGB")
        except:
            init_image = Image.open(settings.image_url).convert("RGB")
        finally:
            if not init_image:
                print("????????????????????????, ?????????????????????????????????")

        init_image = init_image.resize((settings.height, settings.width))
        show_image([init_image], settings.prompt)
 
        for idx in range(settings.sample_num):
            print(f"???????????????{idx+1}????????? - ??????{settings.batch}???")
            if settings.device == "cuda":
                with autocast("cuda"):
                    data = pipe(prompt=[settings.prompt]*settings.batch,
                                image=init_image,
                                strength=settings.strength,
                                num_inference_steps=settings.num_inference_steps,
                                guidance_scale=settings.guidance_scale ,
                                generator=generator)
            else:
                data = pipe(prompt=[settings.prompt]*settings.batch,
                            image=init_image, 
                            strength=settings.strength,
                            num_inference_steps=settings.num_inference_steps,
                            guidance_scale=settings.guidance_scale ,
                            generator=generator)
            image = data["images"]
            save_image(image, settings.save_dir, settings.prompt)
            image_list.append(image_grid(image, x=settings.batch, y=settings.batch))
        show_image(image_list, settings.prompt)
    else:
        pass


if __name__ == "__main__":
    _args = get_args()
    main(_args)

    print("????????????")
    