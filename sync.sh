#!/bin/bash

rsync -az --info=progress2 ast manatees.py data-*.pkl taco@fourier:src/manatees/.

# run on remote:
#   pip install numpy scikit-learn matplotlib timm==0.4.5 wget
#   pip install torch torchaudio torchvision --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121
