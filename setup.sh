#!/bin/bash

apt-get update
apt-get install -y build-essential python3-dev

echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

pip install --upgrade pip
pip install -r requirements.txt

pip list
