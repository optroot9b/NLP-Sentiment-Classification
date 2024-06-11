# NLP Sentiment Classification Project
## Project Overview
This project aims to explore the potential of Natural Language Processing in sentiment classification.
The model is designed to analyse textual data based on the data scraped yelp reviews, with a view to classifying their sentiment as positive or negative (binary classification).
This could be a valuable contribution to a range of applications, including customer feedback analysis and product reviews.

### Upon evaluation of the unseen data (test data), the following results were established.
* **Test Accuracy: 0.8973104953765869**
* **Test F1 Score: 0.9334389857369256**
* **Test ROC AUC Score: 0.9530156277307137**

## Features
* Data Loading and EDA
* Data Preprocessing
* Building Neural Network
* Model Training and Evaluation

## Setup
### SummaryWriter

It initializes a log directory where all the training information will be saved. This information can later be visualized using TensorBoard.

[Tensorboard](https://www.tensorflow.org/tensorboard/scalars_and_keras)

***

### Sentiment Classifier (neural network)

The designed neural network employs vectors of length 384 (embeddings) and generates a binary response (sigmoid) as its output.
There are 3 hidden layers compromising 256, 128, 64 neurons. The activation functions are set to nn.ReLU. The model uses BatchNorm1d normalization.


[BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d)
[ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)
[Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid)
[Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)

***


### loss_function = BCEWithLogitsLoss

This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability. 

[BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)

***

### optimizer = AdamW

AdamW optimizer is used in this project instead of regular adam. Adaptive gradient methods such as Adam may lead to worse generalization. There is a certain inequivalence of L2 regularization and weight decay for Adam. Adam(W) with decoupled weight decay has been observed to yield substantially better generalization performance than the common implementation of Adam with L2 regularization.

[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW)
[arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

***

### scheduler = CosineAnnealingLR

Reduces the learning rate following a cosine decay pattern, which gradually decreases the learning rate to zero.

[CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)

***

## Prerequisites
Ensure you have the following installed on your machine:

* Python 3.7 or higher
* Jupyter Notebook


# Installation
To run this project on your local machine, follow the steps below:

```shell
cd ~
```
### Clone the repository

```shell
git clone https://github.com/optroot9b/nlp_project.git
cd nlp_project

```
### Create a virtual environment

```shell
python3 -m venv nlptask
source nlptask/bin/activate
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Run Jupyter Notebook

```shell
jupyter notebook
```


## Customization

**Model Parameters**: Build your own neural network, test other optimizers, adjust parameters, and experiment with other schedulers. For more information about this topic, please refer to the  [pytorch.optim](https://pytorch.org/docs/stable/optim.html) documentation page.

**Additional Features**: Improve or modify the model by adding more features, such as multiclass classification instead of binary classification, data augmentation as data used in this project seems to be imbalanced. You can try using cross-validation methods such as Leave-One-Out Cross-Validation or K-Fold Cross-Validation, Learning Curves to improve model performance.
