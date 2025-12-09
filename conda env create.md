LLM Environment Setup Guide

This guide provides a step-by-step workflow for setting up a Python environment specifically for LLM (Large Language Model) development on Linux. It covers installing Miniconda, creating a dedicated environment, and installing PyTorch with CUDA 12.4 support.

📋 Prerequisites

OS: Linux (x86_64)

GPU: NVIDIA GPU (for CUDA 12.4 support)

🛠️ Installation Steps

1. Install Miniconda

First, download and install Miniconda to manage your environments. Run the following commands in your terminal:

# Create directory for miniconda
mkdir -p ~/miniconda3

# Download the installation script
wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) -O ~/miniconda3/miniconda.sh

# Run the installation script
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Remove the script to save space
rm ~/miniconda3/miniconda.sh

# Activate the base conda environment
source ~/miniconda3/bin/activate


2. Create the Environment

Create a dedicated Conda environment named llm with Python 3.10.18.

conda create --name llm python==3.10.18 -y


3. Activate the Environment

Crucial Step: Switch to your new environment before installing libraries.

conda activate llm


4. Install PyTorch (CUDA 12.4)

Install PyTorch, Torchvision, and Torchaudio with support for CUDA 12.4.

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)


5. Install Dependencies

Install the remaining project dependencies from your requirements.txt file.

# Ensure you are in the directory containing requirements.txt
pip install -r requirements.txt


⚡ Quick Reference (One-Liner)

If you have already installed Miniconda, you can run this block to set up the rest:

conda create --name llm python==3.10.18 -y && conda activate llm && pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124) && pip install -r requirements.txt
