# **Grid Diffusion Models for Text-to-Video Generation, CVPR 2024**

This repository is the official implementation of [**Grid Diffusion Models for Text-to-Video Generation**](https://arxiv.org/abs/2404.00234).

**[Grid Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2404.00234)**
<br/>
[Taegyeong Lee*](https://sites.google.com/view/taegyeonglee/home), 
[Soyeong Kwon*](https://sites.google.com/view/soyeongkwon),Taehwan Kim

<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://taegyeong-lee.github.io/text2video)
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)](https://arxiv.org/abs/2404.00234)

## Abstract
Recent advances in the diffusion models have significantly improved text-to-image generation. However, generating videos from text is a more challenging task than generating images from text, due to the much larger dataset and higher computational cost required. Most existing video generation methods use either a 3D U-Net architecture that considers the temporal dimension or autoregressive generation. These methods require large datasets and are limited in terms of computational costs compared to text-to-image generation. To tackle these challenges, we propose a simple but effective novel grid diffusion for text-to-video generation without temporal dimension in architecture and a large text-video paired dataset. We can generate a high-quality video using a fixed amount of GPU memory regardless of the number of frames by representing the video as a grid image. Additionally, since our method reduces the dimensions of the video to the dimensions of the image, various image-based methods can be applied to videos, such as text-guided video manipulation from image manipulation. Our proposed method outperforms the existing methods in both quantitative and qualitative evaluations, demonstrating the suitability of our model for real-world video generation.

## News
- [04-02-2024] README.md

## Approach
![image](https://taegyeong-lee.github.io/static/images/t2v/method_overview.png)

## Usage
 We are in the process of refactoring the code.

## Citation
```

```
