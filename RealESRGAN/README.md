###  :book: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

> [[论文](https://arxiv.org/abs/2107.10833)] &emsp; [项目主页] &emsp; [[YouTube 视频](https://www.youtube.com/watch?v=fxHWoDSSvSc)] &emsp; [[B站视频](https://www.bilibili.com/video/BV1H34y1m7sS/)] &emsp; [[Poster](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)] &emsp; [[PPT](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]<br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Tencent ARC Lab; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences


---

我们提供了一套训练好的模型（*RealESRGAN_x4plus.pth*)，可以进行4倍的超分辨率。<br>
**现在的 Real-ESRGAN 还是有几率失败的，因为现实生活的降质过程比较复杂。**<br>
而且，本项目对**人脸以及文字之类**的效果还不是太好，Real-ESRGAN 将会被长期支持，我会在空闲的时间中持续维护更新。

---

## :zap: 快速上手

###       普通图片

下载我们训练好的模型: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
```

推断!

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

结果在`results`文件夹

### 动画图片

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>
有关[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)的更多信息和对比在[**anime_model.md**](docs/anime_model.md)中。
```bash
# 下载模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P experiments/pretrained_models
# 推断
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

结果在`results`文件夹

### Python 脚本的用法

1. 虽然你使用了 X4 模型，但是你可以 **输出任意尺寸比例的图片**，只要实用了 `outscale` 参数. 程序会进一步对模型的输出图像进行缩放。

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Whether to use half precision during inference. Default: False
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

## :european_castle: 模型库

请参见 [src/docs/model_zoo.md](src/docs/model_zoo.md)
