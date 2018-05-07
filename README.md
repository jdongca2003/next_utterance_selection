# Enhanced word representation for Out-of-Vocabulary  on Ubuntu Dialogue Corpus

This is a TensorFlow implementation of the model described in:

Jianxiong Dong, Jim Huang
[Enhance Word Representation For Out-of-Vocabulary on Ubuntu Dialogue Corpus](https://arxiv.org/pdf/1802.02614.pdf).

The model has acheived the state-of-the-art performane on Ubuntu Dialogue Corpus V2 and Douban Chinese dialogue corpus.

## Contact
***Code author:*** Jianxiong Dong


## Contents
* [Requirements](#Requirements)
* [Dataset](#Dataset)
* [Training a model](#Training-a-model)
* [Evaluating a Model](#Evaluating-a-model)

## Requirements

1. Install the Tensorflow library ([instructions](https://www.tensorflow.org/install/)). For example:

```
virtualenv --system-site-packages tensorfow_dev
source tensorflow_dev/bin/activate
pip install --upgrade pip
pip install tensorflow-gpu==1.4.0
```

2. 16GB of RAM. 32GB is recommended.
3. A machine with NVIDIA GPU card (large GPU RAM)  is preferable. It has been tested with NVIDIA Titan Xp (12G GPU RAM).

## Dataset

We used [Ubuntu Dialogue Corpus V2](https://github.com/rkadlec/ubuntu-ranking-dataset-creator). In order to easily reproduce results in the above paper,
the processed dataset has been provided. 


```shell
cd data
sh download.sh

```

## Training a model

Execute the following commands to start the training script. By default it will
run for 230k steps to achieve maximum mean reciprocal rank on the validation set.

```shell
cd bin
nohup sh ubuntu_train.sh &
```

## Evaluating a model
If several runs exist in 'runs' folder,  the checkpoints of the latest run is used to evaluate the model performance.

```shell
cd bin
sh ubuntu_test.sh 
```


