#!/bin/bash

apt-get update
apt-get install -y build-essential python3-dev

pip install --upgrade pip
pip install setuptools wheel

pip install numpy==1.26.4
pip install -r requirements.txt
