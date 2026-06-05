#! /bin/bash

if [ ! -d venv ]; then
  python3 -m venv venv
fi

source venv/bin/activate

pip install -U numpy pandas pillow torch torchinfo torchvision \
        tqdm matplotlib seaborn albumentationsx kaggle kagglehub \
        scikit-learn torchmetrics nibabel Nuitka ninja\
        --upgrade