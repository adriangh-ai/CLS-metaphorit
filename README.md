# CLS-metaphorit

This repository contains the code and results of our project on automatic metaphor detection using the BERT-base-multilingual-cased model. The goal of the project was to classify sentences as metaphorical or non-metaphorical.

## Dataset

We used two datasets for this project, one with single annotator labels and the other with multiple annotators. The datasets contain sentences in Italian and English. The label distribution was skewed towards positive labels in the single annotator dataset and negative labels in the multi-annotator dataset. A subset of the multi-annotator dataset also had multiclass labels for 'migrant', 'covid', or 'none'.

1. **Dataset 1**: This dataset consists of sentences annotated with binary labels for metaphor detection. Each sentence is annotated by a single annotator and the labels are skewed towards positive labels, i.e., the presence of a metaphor.

    | Metaphor | Count |
    |----------|-------|
    | Yes      | 172   |
    | None     | 47    |


2. **Dataset 2**: This dataset is a multi-annotator dataset with sentences annotated by 2 to 5 annotators. The labels in this dataset include binary classifications for metaphor detection as well as multiclass labels indicating the type of metaphor.

    The binary labels are skewed towards negative labels, i.e., the absence of a metaphor, and there are 41 instances where the annotations are evenly split and an agreement cannot be established by majority vote.

    | aggregated_label_binary | Count |
    |-------------------------|-------|
    | No                      | 407   |
    | Yes                     | 156   |
    | Equal split: yes, no    | 41    |


To tackle the metaphor detection task as a binary classification problem, we merge both datasets. This allows us to benefit greatly from each other as the datasets are skewed in different directions. After merging, the distribution of labels is as follows:

| Label | Count |
|-------|-------|
| No    | 454   |
| Yes   | 328   |

For training and evaluating the model, random undersampling was performed to balance the dataset. Furthermore, we use an 80-10-10 random stratified train-dev-test split with a fixed seed for reproducibility.

## Model

We used the 'bert-base-multilingual-cased' model for this project. The model was trained on an 80-10-10 random stratified train-dev-test split of the dataset. We performed hyperparameter search in all runs, and 10 runs with randomly initialized heads.

## Training

The training procedure was performed using HuggingFace Transformers library, and consists of the following steps:

1. **Initialization**: We start with a pre-trained BERT model, specifically 'bert-base-multilingual-cased'. This model has been trained on a large corpus of multilingual data and has learned to generate useful language representations.

2. **Adding a classification head**: On top of the pre-trained BERT model, we add a new classification layer. This head is a simple linear layer followed by a softmax activation function, which is used to output class probabilities.

3. **Fine-tuning**: We fine-tune the entire model (both the pre-trained BERT part and the new classification head) on our specific task using our training dataset.

4. **Hyperparameter search**: We perform a search over possible values of hyperparameters (like learning rate, number of training epochs, batch size, etc.) to find the combination that gives the best performance on our development dataset.

5. **Multiple runs**: To counter variances due to the randomness in neural network training, we repeat the training process with the best hyperparameters 10 times. Each run starts with a different randomly initialized head.

6. **Aggregating results**: At the end, we have 10 slightly different models and their performances on the development dataset. We calculate the average performance and standard deviation across these 10 runs to get a robust estimate of our model's performance.

## Results

The results for the binary classification problem were as follows:

- Undersampling: 
  - F1 score: 76.91
  - Accuracy: 77.95

We have also implemented a method with TF-IDF and a Support Vector Machine classifier, achieving a 75.31 F1 score.

## Future Work

- Extension of the current dataset would significantly aid the learning process of a model and help exploit the strengths of neural networks.
- 'covid' metaphors may be too scarce to obtain a sample comparatively large enough.

