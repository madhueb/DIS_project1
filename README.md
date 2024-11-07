# CS-433 Machine Learning - Project 1
Team : machine-learners
Team members:
- Madeleine HUEBER
- Matin Ansaripour
- Aryan Ahadinia


## Getting started
### Project description
This project is part of the course CS-423 Distributed Information System at EPFL. The goal of the project is to create a multilingual information retrieval system designed to efficiently retrieve the top 10
most relevant documents in response to a user query from a large, diverse corpus.

### Repository structure







### Installation 


To use our model on this project, you will first need to clone the repository 

```bash

!git clone https://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@github.com/madhueb/DIS_project1.git
```

Then, you will need to install the required packages:

- numpy
- pandas
- argparse
- gc
- json
- pickle


## Running the model

To run our model, you will need to run the following command in the terminal:

```bash



```

It will preprocess the data, train the model and output the predictions in the form of a csv file. The predictions will be saved as `submission.csv`.

### Arguments

You can specify the following arguments when running the model:

| Argument              | Description                                    | Type   | Default |
|-----------------------|------------------------------------------------|--------|---------|
| `--seed`              | Set the seed for deterministic results         | `int`  | 42      |
| `--gamma`             | Learning rate for training                     | `float`| 0.1     |
| `--max_iters`         | Maximum number of iterations                   | `int`  | 1000    |
| `--lambda_`           | Regularization parameter                       | `float`| 0       |
| `--undersampling_ratio` | Undersampling ratio to balance the classes    | `float`| 0.2     |

## Parameters exploration

To do some parameters exploration, you can run the following command in the terminal:

```bash

python test.py

```

It will test different values for the hyperparameters listed above and ouput which values of each hyperparameter give the best F-1 score using a 5-fold cross-validation.





