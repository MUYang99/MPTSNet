# MPTSNet
MPTSNet is an implementation of the paper [*MPTSNet: Integrating Multiscale Periodic Local Patterns and Global Dependencies for Multivariate Time Series Classification*] (AAAI 2025).

<p align="center">
<img src="docs/poster.png" width="90%"/>
</p>

## 🛠️ Setup

### Repository

Clone the repository:

```bash
git clone [your-repo-url] && cd MPTSNet
```

### Installation

Create a conda environment with all dependencies:

```bash
conda create --name mptsnet python=3.8
conda activate mptsnet
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

The project supports various UEA time series classification datasets from [Time Series Classification Repository](https://www.timeseriesclassification.com/dataset.php). Place your dataset in the `dataset/General/` and `dataset/UEA/` directory. For quick access to all datasets used in the paper, you can visit [this link](https://drive.google.com/drive/folders/1nV7LjhY2F084jNmE-b9aMEcXNhv6drWG?usp=sharing).

Train MPTSNet:
```bash
python train.py
```

Evaluate MPTSNet:
```bash
python eval.py
```

### Model Configuration

The model automatically adapts its parameters based on input dimensions:
- Embedding dimensions are scaled based on input channels
- Periodic patterns are automatically detected using FFT
- Early stopping and learning rate scheduling are implemented for optimal training

### Dataset Structure

Structure your dataset as follows:
```bash
dataset/General/
└── YOUR_DATASET_NAME
    ├── YOUR_DATASET_NAME_TRAIN.ts
    └── YOUR_DATASET_NAME_TEST.ts
```

## 🏗️ Architecture

MPTSNet consists of several key components:

1. **Data Embedding Layer**
   - Converts input time series into learnable embeddings
   - Adaptive dimension scaling based on input channels

2. **Periodic Block**
   - Inception modules with CBAM attention for local feature extraction
   - Window Transformer for capturing periodic patterns
   - Global Transformer for temporal dependencies

3. **Feature Fusion**
   - FFT-based period importance weights
   - Adaptive feature combination

## 📊 Key Features

- Automatic period detection using FFT
- Multi-scale local feature extraction through inception modules
- Attention-based global feature fusion
- Adaptive embedding dimensions
- Enhanced interpretability

## 📈 Results

Training results are saved in the `results/` directory, including:
- Model checkpoints
- Training logs
- Accuracy metrics

## 🎓 Citation

If you use MPTSNet in your research, please cite our paper:

```bibtex

```

## 📝 License

This project is licensed under the MIT License.
