# Multimodal Semantic Segmentation of Drone Imagery: RGB & Elevation Data Fusion

## 1. Executive Summary
Semantic segmentation of high-resolution aerial imagery is a critical component for topographical mapping, environmental monitoring, and autonomous drone navigation in heavy industries[cite: 1]. However, relying exclusively on RGB optical data often leads to class confusion in complex spatial environments (e.g., distinguishing flat green roofs from raised vegetation)[cite: 1]. 

This research investigates the multimodal fusion of standard RGB imagery with **Digital Surface Models (DSMs / Elevation data)** to enhance predictive spatial accuracy across various machine learning and deep learning architectures[cite: 1]. 

## 2. Project Scope & My Contribution
This repository serves as an academic research archive containing the technical methodologies and evaluation results of a collaborative group project. 

**Personal Contribution:** I was primarily responsible for engineering, training, and evaluating the advanced deep learning architectures within this study, specifically:
*   **DeepLabV3+ (Deep Convolutional Neural Network)**[cite: 1].
*   **SegFormer (Transformer-based Architecture)**[cite: 1].
*   Co-authoring the comprehensive benchmarking report and executive summaries[cite: 1].

*(Note: The implementations of the Random Forest + CRF baseline and U-Net architectures were handled by other project members and are included in the comparative analysis for benchmarking purposes[cite: 1]).*

## 3. Dataset & Challenges
*   **Source:** DroneDeploy Medium Dataset (High-resolution aerial orthophotos)[cite: 1].
*   **Inputs:** 3-channel RGB TIFFs and 1-channel DSM Elevation TIFFs (10 cm per pixel resolution)[cite: 1].
*   **Classes:** 7-class ground truth (Building, Clutter, Vegetation, Water, Ground, Car, Ignore)[cite: 1].
*   **Key Challenges:**
    *   **Class Imbalance:** The dataset is heavily dominated by 'Ground' pixels[cite: 1].
    *   **Spectral Similarity:** Objects with different heights sharing similar textures/colors[cite: 1].
    *   **Computational Limits:** Imagery was processed into 64x64 pixel tiles (6.4m x 6.4m spatial context)[cite: 1].

## 4. Architectural Deep Dive

### 4.1. DeepLabV3+ (DCNN)
To address the variation in object sizes (from large ground areas to small vehicles), the DeepLabV3+ model was engineered with a **ResNet50 backbone**[cite: 1]. 
*   **Key Feature:** Utilized Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale spatial context[cite: 1]. 
*   **Loss Function Optimization:** To mitigate the severe class imbalance, standard cross-entropy was replaced with **Focal Loss**, significantly improving the model's ability to detect minority classes[cite: 1].
*   **Multimodal Input:** Fusing RGB and DSM allowed the network to differentiate classes with similar spectral footprints but distinct physical structures[cite: 1].

### 4.2. SegFormer (Transformer)
Implemented to test whether attention-based feature learning could outperform convolution-based architectures[cite: 1]. 
*   **Architecture:** Utilized a hierarchical transformer encoder coupled with a lightweight MLP decoder to model long-range spatial relationships[cite: 1].
*   **Constraint Discovery:** Transformers generally require massive datasets to learn reliable spatial patterns[cite: 1]. Under the constrained 64x64 tile size and highly imbalanced data, this architecture exhibited significant class-collapse behavior (overpredicting majority classes like 'Building' and 'Ground')[cite: 1].

## 5. Benchmarking & Results

The models were evaluated on an unseen test set using Precision, Recall, and F1-Score metrics[cite: 1].

| Architecture | Precision | Recall | F1 Score | Inference Time |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest + CRF | 0.50 | 0.22 | 0.26 | 6:45.30 |
| U-Net | 0.61 | 0.38 | 0.39 | **1:20.89** |
| **DeepLabV3+** | **0.76** | **0.61** | **0.58** | 2:49.76 |
| SegFormer | 0.21 | 0.12 | 0.15 | 2:33.26 |

**Key Takeaways:**
1.  **Accuracy Winner:** **DeepLabV3+** achieved the best overall performance, providing the optimal balance between correct predictions and class coverage (F1-score: 0.58)[cite: 1]. The convolutional structure effectively preserved local spatial boundaries[cite: 1].
2.  **Transformer Vulnerability:** SegFormer performed the weakest (F1-score: 0.15), proving highly susceptible to overfitting and majority-class collapse when trained on fragmented, small-tile datasets without imbalance-aware loss functions[cite: 1].
3.  **Industrial Applicability:** For deployment scenarios where spatial accuracy and safety are paramount (e.g., mining topography or structural inspections), the DeepLabV3+ model with Focal Loss and multimodal input proves to be the most viable solution[cite: 1].

## 6. Visualizations
*(Insert DeepLabV3+ Confusion Matrix images here)*
