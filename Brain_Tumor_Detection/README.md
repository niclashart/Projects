# Brain Tumor Detection using Deep Learning

A PyTorch-based deep learning project for brain tumor detection and segmentation using convolutional neural networks. This project processes COCO-format annotations to create segmentation masks and trains a CNN model for medical image analysis.

## 🧠 Overview

This project implements an end-to-end pipeline for brain tumor detection that includes:
- Data preprocessing from COCO format annotations
- Mask generation for segmentation tasks
- Custom CNN architecture for image segmentation
- Training pipeline with early stopping and visualization

## 📋 Features

- **COCO Format Support**: Processes COCO JSON annotations for medical images
- **Automatic Mask Generation**: Converts polygon annotations to binary segmentation masks
- **Custom CNN Architecture**: Lightweight segmentation model with 6 convolutional layers
- **Data Visualization**: Tools for visualizing annotations, masks, and training progress
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **GPU Support**: CUDA-compatible training for faster processing

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install torch torchvision
pip install pycocotools
pip install opencv-python
pip install matplotlib
pip install scikit-image
pip install tifffile
pip install scikit-learn
pip install tqdm
pip install Pillow
pip install numpy
```

## 📁 Project Structure

```
Brain_Tumor_Detection/
├── detection.ipynb          # Main notebook with complete pipeline
├── data/
│   ├── train/              # Training images
│   ├── valid/              # Validation images
│   ├── test/               # Test images
│   ├── train2/             # Processed training data
│   │   ├── images/         # Training images
│   │   └── masks/          # Training masks
│   ├── valid2/             # Processed validation data
│   └── test2/              # Processed test data
└── README.md
```

## 🚀 Usage

### 1. Data Preparation

The project expects COCO format annotations. Place your data in the following structure:
```
data/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

### 2. Run the Pipeline

Execute the Jupyter notebook `detection.ipynb` which includes:

1. **Data Loading and Visualization**
2. **Mask Generation from COCO Annotations**
3. **Dataset Creation and Preprocessing**
4. **Model Training**
5. **Results Visualization**

### 3. Key Components

#### Data Processing
```python
# Convert COCO annotations to segmentation masks
main(json_file, mask_output_folder, image_output_folder, original_image_dir)
```

#### Model Architecture
```python
# Custom CNN for segmentation
model = nn.Sequential(
    cnnLayer(C, n_filters),
    *[cnnLayer(n_filters, n_filters) for _ in range(5)],
    nn.Conv2d(n_filters, 1, (3,3), padding=1),
)
```

#### Training
```python
# Train with early stopping
results = Train(model, train_loader, valid_loader, loss_fn, optimizer, early_stopping)
```

## 🔧 Configuration

### Model Parameters
- **Input Channels**: 1 (grayscale)
- **Filters**: 32
- **Architecture**: 6-layer CNN with BatchNorm and LeakyReLU
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: Adam (lr=0.00001)
- **Image Size**: 224x224 pixels

### Training Parameters
- **Batch Size**: 16
- **Early Stopping Patience**: 10 epochs
- **Learning Rate Scheduler**: StepLR (step_size=5, gamma=0.1)

## 📊 Results

The training pipeline provides:
- Training and validation loss curves
- Model checkpointing with early stopping
- Visualization of predictions vs ground truth

## 🔍 Key Functions

- `display_images_with_coco_annotations()`: Visualize COCO annotations
- `create_mask()`: Generate binary masks from polygon annotations
- `CustomDataset_general()`: PyTorch dataset class for image-mask pairs
- `Train()`: Complete training loop with early stopping
- `loss_and_metric_plot()`: Visualize training progress

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## ⚠️ Disclaimer

This project is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and regulatory approval.

## 📧 Contact

For questions or issues, please open an issue on the repository or contact the project maintainer.

---

**Note**: Ensure you have sufficient computational resources (preferably GPU) for training the model, as medical image processing can be computationally intensive.