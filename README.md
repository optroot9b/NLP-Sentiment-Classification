# NLP Sentiment Classification Project
## Project Overview
This project aims to explore the potential of Natural Language Processing in sentiment classification.
The model is designed to analyse textual data based on the yelp reviews dataset, with a view to classifying their sentiment as positive or negative.
This could be a valuable contribution to a range of applications, including **customer feedback analysis and product reviews.**

## Features
* Data Loading and EDA
* Data Preprocessing
* Model Building
* Model Training and Evaluation

## Setup Explanation
* **writer = SummaryWriter()**

It initializes a log directory where all the training information will be saved. This information can later be visualized using TensorBoard.

* **model = ReviewClassifier** 

The designed neural network employs vectors of length 384 (embeddings) and generates a binary response (sigmoid) as its output.
Intermediate layers are configured to comprise 150 and 15 neurons in sequence. The activation functions are set to nn.ReLU. 

* **loss_function = torch.nn.BCEWithLogitsLoss()**

The function is utilized within the code for binary classification tasks, which provide stability in numerical calculations and efficiency in computational processing. This is accomplished through the integration of the sigmoid activation function and binary cross-entropy loss.

* **optimizer = torch.optim.Adam(model.parameters())**

The "Adam" algorithm is an adaptive moment estimation method that combines the advantages of two other extensions of stochastic gradient descent (SGD), namely AdaGrad and RMSProp. Its adaptive learning rate mechanism is used to compute gradients, update bias moments, correct bias moments, and to update parameters.

## Prerequisites
Ensure you have the following installed on your machine:

* Python 3.7 or higher


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

**Model Parameters**: Adjust the model parameters and preprocessing steps as needed to better suit your specific dataset and requirements.

**Additional Features**: Enhance the model by adding more features, such as handling different languages or more complex sentiment classifications.
