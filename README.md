# Multimodal Drone Imagery Semantic Segmentation

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Archive-2E7D32?style=for-the-badge)

## 📖 Executive Summary (About)

This project is a semantic segmentation research and engineering pipeline for high-resolution drone orthophotos, designed to classify every pixel into operationally meaningful land-cover categories such as buildings, vegetation, water, ground, clutter, and cars. The repository implements a Keras/TensorFlow segmentation workflow that converts large aerial scenes into trainable image chips, fuses RGB imagery with Digital Surface Model (DSM) elevation data, trains a deep neural network, runs full-scene inference, and evaluates predictions with weighted precision, recall, F1 score, and confusion matrices.

The technical problem addressed is a common failure mode in aerial intelligence systems: RGB-only imagery can confuse visually similar surfaces, such as flat rooftops and vegetation, when physical structure is not represented. By adding elevation as a fourth modality and using a segmentation-specific training pipeline, the project improves spatial reasoning for use cases such as topographic mapping, environmental monitoring, infrastructure inspection, and autonomous drone analytics.

The documented benchmark summary identifies DeepLabV3+ with RGB+DSM fusion and focal loss as the strongest research configuration, achieving **0.76 weighted precision**, **0.61 weighted recall**, and **0.58 weighted F1 score** on unseen test scenes. The repository also contains an operational Keras U-Net style implementation with mixed-precision training, deterministic scene splits, multimodal chip loading, and full-scene tiled inference.

## 🚀 Technical Highlights & Business Value

- **Multimodal geospatial perception:** Combines 3-channel RGB orthophotos with 1-channel DSM elevation inputs to reduce class confusion in visually ambiguous aerial environments.
- **End-to-end segmentation pipeline:** Automates scene indexing, train/validation/test splitting, chip generation, Keras training, tiled inference, mask reconstruction, and scoring.
- **Imbalance-aware model training:** Uses focal loss to focus optimization on hard and underrepresented classes, a practical requirement for ground-dominated drone imagery.
- **Production-minded evaluation:** Computes weighted precision, recall, F1 score, and normalized confusion matrices, enabling model comparison across Random Forest + CRF, U-Net, DeepLabV3+, and SegFormer baselines.
- **Operational value:** Supports pixel-level mapping workflows for inspection, land-cover analytics, environmental monitoring, and safety-critical site intelligence where boundary accuracy and minority-class recall matter.

## 🛠️ Tech Stack & Skills Demonstrated

* **Languages:** Python
* **Frameworks & Libraries:** TensorFlow, Keras, OpenCV, NumPy, Pillow, Matplotlib, scikit-learn, tqdm, gdown
* **Methodologies:** Semantic Segmentation, Multimodal Data Fusion, Encoder-Decoder CNNs, U-Net Skip Connections, Focal Loss, Mixed-Precision Training, Deterministic Dataset Splitting, Tiled Inference, Weighted Classification Metrics
* **Domain Skills:** Drone Imagery Analytics, Remote Sensing, DSM/Elevation Feature Engineering, Pixel-Level Land-Cover Classification, Computer Vision Model Evaluation, Geospatial Data Pipeline Engineering

## 🔑 Keywords & Tags

semantic segmentation, drone imagery, aerial imagery, remote sensing, geospatial AI, RGB DSM fusion, elevation data, TensorFlow, Keras, U-Net, DeepLabV3+, SegFormer, focal loss, computer vision, land-cover classification

## 🏗️ Architecture & Methodology

The repository is organized around a complete computer vision pipeline for scene-level drone segmentation. The data layer begins with `data/index.csv`, which lists valid scene identifiers. `libs/config.py` applies a deterministic **70/15/15 train-validation-test split** using a fixed random seed, enabling repeatable experiments across the 9-scene sample index and the 55-scene full index.

Large orthophotos, labels, and elevation rasters are converted into fixed-size chips by `libs/images2chips.py`. During chip generation, RGB image tiles, label tiles, and DSM elevation tiles are written into aligned directories. Chips containing the ignore class are skipped to avoid contaminating supervised training with invalid labels. The current configuration uses **128 x 128 pixel chips**, and elevation-aware mode is enabled by default through `USE_ELEVATION = True`.

The model pipeline uses a Keras encoder-decoder architecture with residual convolutional blocks, batch normalization, strided-convolution downsampling, transpose-convolution upsampling, and U-Net style skip concatenations. When elevation is enabled, the model receives two inputs: an RGB tensor and a DSM tensor, which are concatenated channel-wise before feature extraction. `libs/models_keras.py` enables TensorFlow mixed precision globally to improve GPU throughput and memory efficiency during training.

Training is handled by `libs/training_keras.py`, which compiles the model with Adam and a custom serializable focal loss. Dataset loading is implemented through a Keras `Sequence` in `libs/datasets_keras.py`, supporting synchronized augmentation for imagery, elevation, and labels. Image augmentations use bilinear interpolation, while label-safe cloned augmentations use nearest-neighbor interpolation to preserve discrete class IDs.

Inference reconstructs full-scene predictions by sliding over large orthophotos, batching valid chips, predicting per-pixel logits, converting class indices back into color masks, and saving prediction rasters. Scoring then compares predicted masks against ground-truth labels, removes ignored pixels, and reports weighted precision, recall, and F1 score per scene and across the test set.

## 📂 Repository Structure

```text
Drone-Imagery-Semantic-Segmentation/
|-- README.md                         # Project documentation
|-- main_keras.py                     # End-to-end training runner
|-- get_data.py                       # Google Drive data download
|-- image_test.py                     # Chip visualization script
|-- data/
|   |-- index.csv                     # Sample scene index
|   `-- index_full.csv                # Full scene index
`-- libs/
    |-- __init__.py                   # Package initializer
    |-- config.py                     # Labels and split config
    |-- datasets_keras.py             # Keras data sequences
    |-- get_data.py                   # Data download helper
    |-- images2chips.py               # Scene-to-chip conversion
    |-- inference_keras.py            # Tiled inference engine
    |-- models_keras.py               # Model factory wrapper
    |-- models_UNet.py                # U-Net architecture
    |-- scoring.py                    # Metrics and confusion matrix
    `-- training_keras.py             # Training and focal loss
```
