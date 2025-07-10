# Brain Tumor Detection using Deep Learning

A comprehensive deep learning project for detecting brain tumors from MRI images using Convolutional Neural Networks (CNN) and Transfer Learning with ResNet.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements and compares two deep learning approaches for brain tumor detection:

1. **Custom CNN**: A custom convolutional neural network designed specifically for grayscale MRI images
2. **ResNet Transfer Learning**: A ResNet-18 based model leveraging transfer learning for RGB images

The project includes comprehensive data augmentation, model training, evaluation, and interpretability using Grad-CAM visualizations.

## Features

- **Dual Model Approach**: Custom CNN vs ResNet transfer learning
- **Data Augmentation**: Automated image augmentation to improve model generalization
- **Grad-CAM Visualization**: Interpretable AI with tumor region highlighting
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score
- **GPU Acceleration**: Optimized for NVIDIA RTX 4060 with CUDA support
- **Jupyter Notebooks**: Interactive development and analysis

## Dataset

The project uses a brain MRI dataset with two classes:
- **Tumor**: MRI images containing brain tumors
- **No Tumor**: MRI images without brain tumors

### Data Processing
- Images resized to 240x240 pixels
- Custom CNN: Grayscale conversion
- ResNet: RGB format
- Data augmentation applied for class balance

## Model Architectures

### Custom CNN
- 4 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
- Global average pooling
- Fully connected layer with sigmoid activation
- Designed for grayscale input

### ResNetCAM (Transfer Learning)
- Based on pretrained ResNet-18
- Modified final layer for binary classification
- Global average pooling with sigmoid activation
- Designed for RGB input
- Grad-CAM support for localization

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Install PyTorch with CUDA support**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install other dependencies**
   ```bash
   pip install -r requirements_pytorch.txt
   ```

4. **Verify installation**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```

## Usage

### Data Preparation
1. Place your brain MRI dataset in the `Brain-Tumor-Detection-Dataset/` directory
2. Organize images into `yes/` (tumor) and `no/` (no tumor) subdirectories
3. Run data augmentation notebook if needed

### Training Models

#### Custom CNN
```bash
jupyter notebook Custom_CNN.ipynb
```
- Follow the notebook cells to train the custom CNN
- Model will be saved as `best_custom_cnn.pth`

#### ResNet Model
```bash
jupyter notebook ResNet_Model.ipynb
```
- Follow the notebook cells to train the ResNet model
- Model will be saved as `best_brain_tumor_model.pth`

### Data Augmentation
```bash
jupyter notebook Augmentation_Dataset.ipynb
```
- Apply data augmentation techniques
- Generate balanced dataset

## Results

### Performance Comparison

| Model | Accuracy | F1-Score | Precision | Recall | Specificity |
|-------|----------|----------|-----------|--------|-------------|
| Custom CNN | 96.22% | 0.9615 | 0.9793 | 0.9444 | 0.9800 |
| ResNetCAM | 98.67% | 0.9870 | 0.9870 | 0.9870 | 0.9863 |

### Key Findings
- **ResNetCAM outperforms Custom CNN** in all metrics
- **Better localization** with Grad-CAM visualizations
- **More stable training** with transfer learning
- **Higher sensitivity** for tumor detection

### Confusion Matrix (ResNetCAM)
- True Negatives: 433
- False Positives: 6
- False Negatives: 6
- True Positives: 455

## Project Structure

```
Brain_Tumor_detection_IA/
├── README.md                           # This file
├── requirements_pytorch.txt            # Python dependencies
├── Brain_Tumor_Detection_Project_Report.txt  # Detailed project report
├── Custom_CNN.ipynb                    # Custom CNN implementation
├── ResNet_Model.ipynb                  # ResNet transfer learning
├── Augmentation_Dataset.ipynb          # Data augmentation pipeline
├── best_custom_cnn.pth                 # Trained custom CNN model
├── best_brain_tumor_model.pth          # Trained ResNet model
├── Brain-Tumor-Detection-Dataset/      # Original dataset
│   ├── yes/                           # Tumor images
│   └── no/                            # No tumor images
├── augmented_data/                     # Augmented images
│   ├── yes/                           # Augmented tumor images
│   └── no/                            # Augmented no tumor images
└── combined_data/                      # Combined original + augmented
    ├── yes/                           # Combined tumor images
    └── no/                            # Combined no tumor images
```

## Technical Details

### Training Configuration
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam with learning rate scheduling
- **Hardware**: NVIDIA RTX 4060 with CUDA 12.1
- **Image Size**: 240x240 pixels
- **Batch Size**: Optimized for GPU memory

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1-Score**: Harmonic mean of precision and recall

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines
- Follow PEP 8 style guidelines
- Add comments to complex code sections
- Update documentation for new features
- Test your changes thoroughly


---

**Note**: This project is for educational and research purposes. For clinical applications, please consult with medical professionals and ensure proper validation protocols are followed. 