#! /bin/bash

if [ ! -d venv ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install -U numpy pandas torch torchvision tqdm matplotlib torchmetrics seaborn torchinfo scikit-learn albumentations kaggle kagglehub