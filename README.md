# TRiCF-Net
This repo is the official implementation for TRiCF-Net: A Thermal Radiation Perception
and Coordinate-Guided Cross-Modal Object Detection Network

# Overview
## Framework
<img width="1045" height="358" alt="绘图2" src="https://github.com/user-attachments/assets/4b853bb6-84dd-4344-8f59-bc37e3492b87" />


## Visualization
<img width="457" height="134" alt="绘图7" src="https://github.com/user-attachments/assets/15f722b9-215d-42ce-9044-bca54d20fed7" />


# Installation
## Clone the repository
git clone [https://github.com/0aaaa0/TRiCF.git](https://github.com/0aaaa0/TRiCF.git)

cd TRiCF
## Create Environment
```shell
conda create -n TRiCF python=3.8
conda activate TRiCF
pip install -r requirements.txt
```
# Datasets 
## Datasets
"FLIR-aligned"  "VEDAI"   "M3FD"
Please download the datasets from the following links and place them in `data/multispectral/`

# Manual Training & Evaluation
## Training
```shell
python train.py --data data/multispectral/FLIR-align-3class.yaml 
```
## Evaluation
```shell
python test.py --data data/multispectral/FLIR-align-3class.yaml --weights best.pt
```
