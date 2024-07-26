#!/bin/bash

# Install ta-lib using conda
wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc
conda create -n talib-uk python=3.9 -y
conda activate talib-uk
conda install -c conda-forge ta-lib -y

# Install other dependencies using pip
pip install -r requirements.txt
