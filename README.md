# Enhanced word representation for Out-of-Vocabulary  on Ubuntu Dialogue Corpus

This is a TensorFlow implementation of the model described in:

Jianxiong Dong, Jim Huang
[Enhance Word Representation For Out-of-Vocabulary on Ubuntu Dialogue Corpus](https://arxiv.org/pdf/1802.02614.pdf).


## Contact
***Code author:*** Jianxiong Dong (github: jdongca2003)


## Contents
* [Getting Started](#getting-started)
    * [Install Required Packages](#install-required-packages)
* [Dataset](#Dataset)
* [Training a model](#Training-a-model)
* [Evaluating a Model](#Evaluating-a-model)

## Getting Started

### Install Required Packages
First ensure that you have installed the following required packages:

* **TensorFlow-gpu 1.4.0** ([instructions](https://www.tensorflow.org/install/))


## Dataset

We used [Ubuntu Dialogue Corpus V2](https://github.com/rkadlec/ubuntu-ranking-dataset-creator). In order to easily reproduce results in the above paper,
the processed dataset has been provided. 


```shell
cd next_utterance_selection/data
sh download.sh

```

## Training a model

Execute the following commands to start the training script. By default it will
run for 230k steps to achieve maximum MRR on the validation set.

```shell
cd next_utterance_selection/bin
nohup sh ubuntu_train.sh &
```

## Evaluating a model
If several runs exist in 'runs' folder,  the checkpoints of the latest run is used to evaluate the model performance.

```shell
cd next_utterance_selection/bin
sh ubuntu_test.sh 
```


