# PanoStyle: Semantic, Geometry-Aware and Shading Independent Photorealistic Style Transfer for Indoor Panoramic Scenes(ICCV 2023)

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.2.0](https://img.shields.io/badge/pytorch-1.2.0-green.svg?style=plastic)
![pyqt5 5.13.0](https://img.shields.io/badge/pyqt5-5.13.0-green.svg?style=plastic)

![image](./docs/assets/Teaser.png)
<!-- **Figure:** *Face image editing controlled via style images and segmentation masks with SEAN* -->

While current style transfer models have achieved impressive results for the application of artistic style to generic images, they face challenges in achieving photorealistic performances on indoor scenes, especially the ones represented by panoramic images.  Moreover, existing models overlook the unique characteristics of indoor panoramas, which possess particular geometry and semantic properties. To address these limitations, we propose the first geometry-aware and shading-independent, photorealistic and semantic style transfer method for indoor panoramic scenes. Our approach extends semantic-aware generative adversarial architecture capabilities by introducing two novel strategies to account the geometric characteristics of indoor scenes and to enhance performance. Firstly, we incorporate strong geometry losses that use layout and  depth inference at the training stage to enforce shape consistency between generated and ground truth scenes. Secondly, we apply a shading decomposition scheme to extract the albedo and normalized shading signal from the original scenes, and we apply the style transfer on albedo instead of full RGB images, thereby preventing shading-related bleeding issues. On top of that, we apply super-resolution to the resulting scenes to improve image quality and yield fine details. We evaluate our model's performance on public domain synthetic data sets. Our proposed architecture outperforms state-of-the-art style transfer models in terms of perceptual and accuracy metrics, achieving a 26.76\% lower ArtFID, a 6.95\% higher PSNR, and a 25.23\% higher SSIM. The visual results show that our method is effective in producing realistic and visually pleasing indoor scenes.

> **PanoStyle: Semantic, Geometry-Aware and Shading Independent Photorealistic Style Transfer for Indoor Panoramic Scenes** <br>
> Tukur, M and Rehman, A Ur and Pintore, G and Gobbetti, E and Schneider, J and Agus, M <br>
> *IEEE/CVF International Conference on Computer Vision **ICCV 2023***


[[Paper](https://openaccess.thecvf.com/content/ICCV2023W/CVAAD/papers/Tukur_PanoStyle_Semantic_Geometry-Aware_and_Shading_Independent_Photorealistic_Style_Transfer_for_ICCVW_2023_paper.pdf)]
[[Project Page](https://github.com/abumuzzammil90/PanoStyle)]
[[Demo](https://www.youtube.com/watch?v=yxTG1GfW5cE&t=2s)]

 
## Installation
Clone this repo.
```bash
Under Construction..
Please check back later. 
```

<!--
Clone this repo.
```bash
git clone https://github.com/abumuzzammil90/PanoStyle.git
cd PanoStyle/
```

This code requires PyTorch, python 3+ and Pyqt5. Please install dependencies by
```bash
pip install -r requirements.txt
```

This model requires a lot of memory and time to train. To speed up the training, we recommend using 4 V100 GPUs


## Dataset Preparation

This code uses [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) and [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset. The prepared dataset can be directly downloaded [here](https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view?usp=sharing). After unzipping, put the entire CelebA-HQ folder in the datasets folder. The complete directory should look like `./datasets/CelebA-HQ/train/` and `./datasets/CelebA-HQ/test/`.


## Generating Images Using Pretrained Models

Once the dataset is prepared, the reconstruction results be got using pretrained models.


1. Create `./checkpoints/` in the main folder and download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/1UMgKGdVqlulfgOBV4Z0ajEwPdgt3_EDK/view?usp=sharing). Save the tar in `./checkpoints/`, then run

    ```
    cd checkpoints
    tar CelebA-HQ_pretrained.tar.gz
    cd ../
    ```

2. Generate the reconstruction results using the pretrained model.
	```bash
   python test.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0
    ```

3. The reconstruction images are saved at `./results/CelebA-HQ_pretrained/` and the corresponding style codes are stored at `./styles_test/style_codes/`.

4. Pre-calculate the mean style codes for the UI mode. The mean style codes can be found at `./styles_test/mean_style_code/`.

	```bash
    python calculate_mean_style_code.py
    ```


## Training New Models

To train the new model, you need to specify the option `--dataset_mode custom`, along with `--label_dir [path_to_labels] --image_dir [path_to_images]`. You also need to specify options such as `--label_nc` for the number of label classes in the dataset, and `--no_instance` to denote the dataset doesn't have instance maps.


```bash
python train.py --name [experiment_name] --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/train/labels --image_dir datasets/CelebA-HQ/train/images --label_nc 19 --no_instance --batchSize 32 --gpu_ids 0,1,2,3
```

If you only have single GPU with small memory, please use `--batchSize 2 --gpu_ids 0`.


## UI Introduction

We provide a convenient UI for the users to do some extension works. To run the UI mode, you need to:

1. run the step **Generating Images Using Pretrained Models** to save the style codes of the test images and the mean style codes. Or you can directly download the style codes from [here](https://drive.google.com/file/d/153U5q_CfwPM0V4wRP199BhD9niUuVW95/view?usp=sharing). (Note: if you directly use the downloaded style codes, you have to use the pretrained model.

2. Put the visualization images of the labels used for generating in `./imgs/colormaps/` and the style images in `./imgs/style_imgs_test/`. Some example images are provided in these 2 folders. Note: the visualization image and the style image should be picked from `./datasets/CelebAMask-HQ/test/vis/` and `./datasets/CelebAMask-HQ/test/labels/`, because only the style codes of the test images are saved in `./styles_test/style_codes/`. If you want to use your own images, please prepare the images, labels and visualization of the labels in `./datasets/CelebAMask-HQ/test/` with the same format, and calculate the corresponding style codes.

3. Run the UI mode

    ```bash
    python run_UI.py --name CelebA-HQ_pretrained --load_size 256 --crop_size 256 --dataset_mode custom --label_dir datasets/CelebA-HQ/test/labels --image_dir datasets/CelebA-HQ/test/images --label_nc 19 --no_instance --gpu_ids 0
    ```
4. How to use the UI. Please check the detail usage of the UI from our [Video](https://youtu.be/0Vbj9xFgoUw).

	[![image](./docs/assets/UI.png)](https://youtu.be/0Vbj9xFgoUw)

## Other Datasets
Will be released soon.

## License

All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**) The code is released for academic research use only.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{tukur2023panostyle,
  title={PanoStyle: Semantic, Geometry-Aware and Shading Independent Photorealistic Style Transfer for Indoor Panoramic Scenes},
  author={Tukur, M and Rehman, A Ur and Pintore, G and Gobbetti, E and Schneider, J and Agus, M},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1553--1564},
  year={2023}
}
```

## Acknowledgments
<-- We thank Wamiq Reyaz Para for helpful comments. This code borrows heavily from SPADE. We thank Taesung Park for sharing his codes. This work was supported by the KAUST Office of Sponsored Research (OSR) under AwardNo. OSR-CRG2018-3730.
We appreciate the previous open-source work - SEAN.  
