#!/bin/bash
set -e

# System dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Clone repo
git clone https://github.com/SigmaWave/eib.git
cd eib
mkdir -p database

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Ollama + model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b