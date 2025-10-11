# Lightweight head for semantic segmentation using DINOv3 as backbone

This repository provides a lightweight object detection head designed to run on top of Meta’s [DINOv3](https://github.com/facebookresearch/dinov3) backbone. The model combines a projection layer with a lightweight ASPP-style decoder, enhanced by depthwise separable convolutions, to efficiently generate dense semantic predictions from DINOv3 features while keeping the parameter count low. It has been trained using the [COCO dataset](https://cocodataset.org/) with panoptic annotations.

This head is part of the [dinov3_ros](https://github.com/Raessan/dinov3_ros) project, where it enables real-time semantic segmentation in ROS 2 by reusing backbone features across multiple perception tasks.


## Table of Contents

1. [Installation](#installation)
2. [Model and loss function](#model_loss)
3. [Usage](#usage)
4. [Integration with dinov3_ros](#integration_dinov3_ros)
5. [Demo](#demo)
6. [License](#license)
7. [References](#references)


## Installation

We recommend using a fresh `conda` environment to keep dependencies isolated. DINOv3 requires Python 3.11, so we set that explicitly.

```
conda create -n semantic_segmentation_dinov3 python=3.11
conda activate semantic_segmentation_dinov3
git clone --recurse-submodules https://github.com/Raessan/semantic_segmentation_dinov3
cd semantic_segmentation_dinov3
pip install -e .
```

The only package that has to be installed separately is PyTorch, due to its dependence with the CUDA version. For example:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129 
```

Finally, we provide weights for the lightweight heads developed by us in the `weights` folder, but the DINOv3 backbone weights should be requested and obtained from their [repo](https://github.com/facebookresearch/dinov3). Its default placement is in `dinov3_weights` folder. The presented head has been trained using the `vits16plus` model from DINOv3 as a backbone.

Training also requires the [COCO dataset](https://cocodataset.org/) with panoptic annotations to be installed on disk.

## Model and loss function

This repository implements a lightweight semantic segmentation head that can be attached to the [DINOv3](https://github.com/facebookresearch/dinov3) backbone (or any ViT producing a single spatial feature map). The design has three main components:

### Model architecture

#### ASPP-style Decoder

- Expands DINOv3 features into a higher-dimensional space for richer representations.

- Uses multiple parallel 3×3 convolutions (simulating a lightweight Atrous Spatial Pyramid Pooling module) to capture multi-scale context.

- Concatenates and fuses these multi-scale features through a projection layer.

- Adds a depthwise separable convolution block for efficiency before the final classifier.

- Outputs per-pixel class logits, upsampled to the input resolution.

### Loss functions

Training uses a combination of two complementary losses:

#### Focal loss (Softmax-based)

- Reduces the impact of easy pixels and focuses training on hard-to-classify regions, improving performance on imbalanced classes.

#### Dice loss (Multiclass)

- Optimizes overlap between predicted masks and ground-truth masks, encouraging accurate boundaries and handling class imbalance.

The total loss is a weighted sum:

$$
L_{total} = \lambda_{focal} L_{focal} + \lambda_{dice} L_{dice}
$$

## Usage

There are three main folders and files that the user should use:

`config/config.py`: This file allows the user to configure the model, loss, training step or inference step. The parameters are described in the file.

`train/train_segmentor.ipynb`: Jupyter notebook for training the segmentor. It can load and/or save checkpoints depending on the configuration in `config.py`.

`inference/inference.py`: Script for running inference with a trained model on new images.

Additionally, the repository includes a `src` folder that contains the backend components: dataset utilities, backbone/head model definitions, and helper scripts. In particular:

- `common.py`: general-purpose functions that can be reused across different task-specific heads.
- `utils.py`: utilities tailored specifically for semantic segmentation (e.g., generate the segmentation overlay).

The segmentor was trained for a total of 30 epochs: first for 15 epochs with a learning rate of 1e-4 using data augmentation, followed by 15 epochs with a reduced learning rate of 1e-5 without augmentation. The final weights have been placed in the `weights` folder.

Our main objective was not to surpass state-of-the-art models, but to train a head with solid results that enables collaboration and contributes to building a more refined [dinov3_ros](https://github.com/Raessan/dinov3_ros). This effort is particularly important because Meta has not released lightweight task-specific heads. For this reason, we welcome contributions — whether it’s improving this segmentation head, adding new features, or experimenting with alternative model architectures. Feel free to open an issue or submit a pull request! See the [Integration with dinov3_ros](#integration-dinov3_ros) section to be compliant with the [dinov3_ros](https://github.com/Raessan/dinov3_ros) project. 

## Integration with [dinov3_ros](https://github.com/Raessan/dinov3_ros)

This repository is designed to be easily integrated into [dinov3_ros](https://github.com/Raessan/dinov3_ros). To enable plug-and-play usage, the following files must be exported from the `src` folder to the `dinov3_toolkit/head_segmentation` folder in [dinov3_ros](https://github.com/Raessan/dinov3_ros):

- `model_head.py`: defines the detection head architecture.
- `utils.py`: task-specific utilites for object detection.
- `class_names.txt`: mapping from class indices to human-readable labels.

Additionally, we provide our chosen weights in `weights/model.pth`.

Any modification or extension of this repository should maintain these files and remain self-contained, so that the head can be directly plugged into [dinov3_ros](https://github.com/Raessan/dinov3_ros) without additional dependencies.

**Update**: To run with `dinov3_ros_tensorrt`, we created an onnx exporter inside `inference/export_model.py`. The resulting `.onnx` models are required instead of the `model_head.py` and `weights/model.pth` to run with TensorRT.

## Demo

<img src="assets/gif_semantic_segmentation.gif" height="800">

### Note on visualization

The labels in the visualizations are placed at the centroid of all pixels belonging to a class. This means that for classes with multiple disconnected regions or multiple instances, the label may appear outside of the actual objects. While this does not affect the predictions themselves, it is important to keep in mind when interpreting the plots.

## License
- Code in this repo: Apache-2.0.
- DINOv3 submodule: licensed separately by Meta (see its LICENSE).
- We don't distribute DINO weights. Follow upstream instructions to obtain them.

## References

- [Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski (2025). Dinov3. *arXiv preprint arXiv:2508.10104.*](https://github.com/facebookresearch/dinov3)

- [Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár (2014). Microsoft COCO: Common Objects in Context. *European conference on computer vision*](https://cocodataset.org)

- [Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille (2016). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE transactions on pattern analysis and machine intelligence*](https://ieeexplore.ieee.org/abstract/document/7913730)

- [Escarabajal, Rafael J. (2025). dinov3_ros](https://github.com/Raessan/dinov3_ros)