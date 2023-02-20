<p align="center">
  <img src="./assets/gfpgan_logo.png" height=130>
</p>

## 项目使用示例

```shell
python demo.py -i inputs/whole_imgs -o results -v 1.3 -s 2
Usage: python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2 [options]...

  -h                   show this help
  -i input             Input image or folder. Default: inputs/whole_imgs
  -o output            Output folder. Default: results
  -v version           GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3
  -s upscale           The final upsampling scale of the image. Default: 2
  -bg_upsampler        background upsampler. Default: realesrgan
  -bg_tile             Tile size for background sampler, 0 for no tile during testing. 		Default: 400
  -suffix              Suffix of the restored faces
  -only_center_face    Only restore the center face
  -aligned             Input are aligned faces
  -ext                 Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```



## 不同型号版本的比较

- **注意V1.3并不总是比V1.2好，可能需要根据您的目的和输入尝试不同的模型**

| Version |                          Strengths                           |                        Weaknesses                        |
| :-----: | :----------------------------------------------------------: | :------------------------------------------------------: |
|  V1.3   | ✓ natural outputs<br> ✓better results on very low-quality inputs <br> ✓ work on relatively high-quality inputs <br>✓ can have repeated (twice) restorations | ✗ not very sharp <br> ✗ have a slight change on identity |
|  V1.2   |          ✓ sharper output <br> ✓ with beauty makeup          |               ✗ some outputs are unnatural               |

- [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth) with colorization and need cuda 
  support
- [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth)
- [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) 

For the following images, you may need to **zoom in** for comparing details, or **click the image** to see in the full size.

|                            Input                             |                              V1                              |                             V1.2                             |                             V1.3                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![019_Anne_Hathaway_01_00](assets/153762146-96b25999-4ddd-42a5-a3fe-bb90565f4c4f.png) | ![](assets/153762256-ef41e749-5a27-495c-8a9c-d8403be55869.png) | ![](assets/153762297-d41582fc-6253-4e7e-a1ce-4dc237ae3bf3.png) | ![](assets/153762215-e0535e94-b5ba-426e-97b5-35c00873604d.png) |
| ![106_Harry_Styles_00_00](assets/153789040-632c0eda-c15a-43e9-a63c-9ead64f92d4a.png) | ![](assets/153789172-93cd4980-5318-4633-a07e-1c8f8064ff89.png) | ![](assets/153789185-f7b268a7-d1db-47b0-ae4a-335e5d657a18.png) | ![](assets/153789198-7c7f3bca-0ef0-4494-92f0-20aa6f7d7464.png) |
| ![076_Paris_Hilton_00_00](assets/153789607-86387770-9db8-441f-b08a-c9679b121b85.png) | ![](assets/153789619-e56b438a-78a0-425d-8f44-ec4692a43dda.png) | ![](assets/153789633-5b28f778-3b7f-4e08-8a1d-740ca6e82d8a.png) | ![](assets/153789645-bc623f21-b32d-4fc3-bfe9-61203407a180.png) |
| ![008_George_Clooney_00_00](assets/153790017-0c3ca94d-1c9d-4a0e-b539-ab12d4da98ff.png) | ![](assets/153790028-fb0d38ab-399d-4a30-8154-2dcd72ca90e8.png) | ![](assets/153790044-1ef68e34-6120-4439-a5d9-0b6cdbe9c3d0.png) | ![](assets/153790059-a8d3cece-8989-4e9a-9ffe-903e1690cfd6.png) |
| ![057_Madonna_01_00](assets/153790624-2d0751d0-8fb4-4806-be9d-71b833c2c226.png) | ![](assets/153790639-7eb870e5-26b2-41dc-b139-b698bb40e6e6.png) | ![](assets/153790651-86899b7a-a1b6-4242-9e8a-77b462004998.png) | ![](assets/153790655-c8f6c25b-9b4e-4633-b16f-c43da86cff8f.png) |
| ![044_Amy_Schumer_01_00](assets/153790811-3fb4fc46-5b4f-45fe-8fcb-a128de2bfa60.png) | ![](assets/153790817-d45aa4ff-bfc4-4163-b462-75eef9426fab.png) | ![](assets/153790824-5f93c3a0-fe5a-42f6-8b4b-5a5de8cd0ac3.png) | ![](assets/153790835-0edf9944-05c7-41c4-8581-4dc5ffc56c9d.png) |
| ![012_Jackie_Chan_01_00](assets/153791176-737b016a-e94f-4898-8db7-43e7762141c9.png) | ![](assets/153791183-2f25a723-56bf-4cd5-aafe-a35513a6d1c5.png) | ![](assets/153791194-93416cf9-2b58-4e70-b806-27e14c58d4fd.png) | ![](assets/153791202-aa98659c-b702-4bce-9c47-a2fa5eccc5ae.png) |
