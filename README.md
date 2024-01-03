# GRL Exam Research Project

This is the official repository for the research project, as part of the GRL exam, 
where we are investigating the following question:

*How does the addition of a global readout affect the expressive power of the Graph Isomorphism
Network in node classification?*

## Project Overview

In this study we introduced a global readout to the standard GIN model, aligning it more
closely with the folklore 1-WL test, which is anticipated to enhance its expressive power, 
especially in node-level tasks. Our empirical evaluation, conducted on two distinct graph datasets,
confirmed these theoretical insights, as we observed modest improvements
in the GIN-GR model’s performance.

## Getting Started

Follow these instructions to replicate our study's findings.

### Dependencies

System Requirements:
- Python version: 3.8.2
- No GPU needed for computations.

### Installation Guide

To set up the project environment: 

```bash
git clone https://github.com/123anonymous321/Exam
```

```bash
cd Exam
```

```bash
pip install -r requirements.txt
```

### Reproduce Results
You have two options to replicate our results: running only the evaluation, or performing both the training and evaluation.

For Training:
Execute the following commands for the respective datasets:

```bash
python src/train_node_classification.py --dataset cora
```
```bash
python src/train_node_classification.py --dataset citeseer
```

For Evaluation:
To evaluate the models with the final hyperparameters as detailed in our report:

```bash
python src/evaluate.py
```
