# CRISP
CRISP is developed for predicting drug perturbation response for unseen cell types in cell type-specific way. It incorporates single cell foundation models (scFMs) into a cell type-specific learning framework. By exploiting cell type similarities and divergences learned from FMs, CRISP effectively extends existing cell atlases into the perturbation space, providing a systematic approach to characterize drug-induced cellular state transitions. In the inference stage, CRISP requires only drug information and control state scRNA-seq data as input, to predict the cell type-specific drug responses.

## Installation

```bash
git clone https://github.com/Susanxuan/CRISP.git
cd CRISP
pip install -e .
```

## Quick Start

Training: \
Follow the [tutorial notebook for training](/tutorials/training.ipynb)

Prediction with trained model: \
Follow the [tutorial notebook](/tutorials/zeroshot_prediction.ipynb). We provide the trained model parameter from Neurips and Sciplex3, repectively.

## Data

Preprocessed datasets used in this work all can be downloaded [here](https://drive.google.com/drive/folders/1QWjmpYZMaqxfLwIeLjwoz-H9vX60udeu?usp=drive_link). There are four perturbation datasets: NeurIPS, SciPlex3, GBM, PC9, and one normal dataset PBMC-Bench. The code of data preprocessing is provided in [data folder](data/)






