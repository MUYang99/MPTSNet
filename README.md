# MPTSNet
MPTSNet is an implementation of the paper [*MPTSNet: Integrating Multiscale Periodic Local Patterns and Global Dependencies for Multivariate Time Series Classification*] (AAAI 2025).

<p align="center">
<img src="docs/poster-compressed.png" width="90%"/>
</p>

## 🛠️ Setup

### Repository

Clone the repository:

```bash
git clone https://github.com/MUYang99/MPTSNet.git && cd MPTSNet
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

## 📈 Results

Training results are saved in the `results/` directory, including:
- Model checkpoints
- Training logs
- Accuracy metrics

## 🎓 Citation

If you find it's useful, please cite our paper:

```bibtex
@inproceedings{mu2025mptsnet,
  title={MPTSNet: Integrating Multiscale Periodic Local Patterns and Global Dependencies for Multivariate Time Series Classification},
  author={Mu, Yang and Shahzad, Muhammad and Zhu, Xiao Xiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={18},
  pages={19572--19580},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License.
