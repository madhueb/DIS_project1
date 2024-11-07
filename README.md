# CS-433 Machine Learning - Project 1
Team : machine-learners
Team members:
- Madeleine HUEBER
- Matin Ansaripour
- Aryan Ahadinia


## Getting started
### Project description
This project is part of the course CS-423 Distributed Information System at EPFL. The goal of the project is to create a multilingual information retrieval system designed to efficiently retrieve the top 10 most relevant documents in response to a user query from a large, diverse corpus. To do so we implemented four different models : A DPR model, a BM25 model, a TF-IDF model and comined model of TF-IDF and BM25. 

### Repository structure
repository_root/ │ ├── scripts/ # Contains all the scripts for running experiments and evaluations │ ├── bm25_tfidf/ # Scripts for BM25 and TF-IDF indexing and retrieval │ └── dpr/ # Scripts for DPR (Dense Passage Retrieval) model training and inference │ ├── src/ # Main source code directory │ ├── bm25_tfidf/ # Implementation of BM25 and TF-IDF models and utility functions │ ├── dpr/ # Implementation of DPR models, encoders, and related modules │ └── init.py # Makes the src directory a Python package │ ├── README.md # Documentation and overview of the project ├── requirements.txt # List of dependencies needed to run the project ├── setup.py # Setup script for package installation





### Installation 


To use our model on this project, you will first need to clone the repository 

```bash

!git clone https://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@github.com/madhueb/DIS_project1.git
```

Then, you will need to install the required packages. You can do this by running the following command in the terminal:

```bash

pip install -r requirements.txt
```


## Running the model

To run our model, you will need to run the following command in the terminal:

```bash



```


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





