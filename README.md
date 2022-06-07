# AutoWS-Bench-101

## Introduction
**AutoWS-Bench-101** is a framework for evaluating automated WS (AutoWS) techniques in challenging WS settings---a set of diverse application domains on which it has been previously difficult or impossible to apply traditional WS techniques.

## Installation
Install anaconda: Instructions here: https://www.anaconda.com/download/  

Clone the repository:
```
https://github.com/Kaylee0501/FWRENCH.git
cd FWRENCH
```
Create virtual environment:
```
conda env create -f env_new_new.yml
source activate FWRENCH
```
## Datasets
Our benchmark auotomatic download the dataset for you. Please run [`data_settings.py`](https://github.com/Kaylee0501/FWRENCH/blob/main/fwrench/utils/data_settings.py) to download the specific dataset you need.

| Name           | # class       | # train      |
| -------------- | ------------- |------------- |
| MNIST          | 10            | 57000
| FashionMNIST   | 10            | 57000
| KMNIST         | 10            | 57000
| CIFAR10        | 10            | 
| SphericalMNIST | 10            |
| PermutedMNIST  | 10            |
| ECG            | 4             |
| EMBER          | 2             |
| NavierStokes   | 2             |
