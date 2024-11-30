# Predicting Co-word Links via Heterogeneous Graph Convolutional Networks

This repository contains the code and resources for the research presented in the paper **"Predicting Co-word Links via Heterogeneous Graph Convolutional Networks"**. The research explores a novel approach to co-word analysis using a heterogeneous graph convolutional network (GCN) to predict potential links between co-words in scientific literature. The approach improves upon traditional machine learning methods by jointly learning word embeddings and document embeddings, thus enhancing link prediction performance.

## Research Overview

Co-word analysis is a technique for discovering research themes by analyzing the co-occurrence of key terms within a field. This study leverages machine learning models, particularly Graph Convolutional Networks (GCNs), to predict links in co-word networks, providing insights into the relationships between research themes and emerging trends in scientific domains.

Key contributions of this work:
- Introduces a heterogeneous graph model combining word co-occurrence and word-document relations.
- Proposes a GCN-based method for link prediction in co-word networks, outperforming traditional machine learning models such as XGBoost and SVM.
- Demonstrates the effectiveness of this approach on the Web of Science dataset from Information Science and Library Science.

## Requirements

This project requires the following Python packages:
- `dgl==2.1.0+cu118`
- `lda==3.0.2`
- `nltk==3.8.1` 
- `numpy==2.1.3`
- `pandas==2.2.3`
- `scikit_learn==1.3.0`
- `scipy==1.14.1`
- `torch==2.2.0`
- `xgboost==2.1.3`
- `torch_geometric==2.6.1`
- `tqdm==4.65.0`
- `yake==0.4.8`

You can install the required dependencies by running:
```bash
pip install -r requirements.txt
```
## Installation
1.Clone this repository:
```bash
git clone https://github.com/66louislee66/co-word-link-prediction.git
cd co-word-link-prediction
```
2.Create and activate a virtual environment (optional but recommended):
```bash
conda create --name dglenv python=3.10
conda activate dglenv
```
3.Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Dataset
The experiments were conducted on the Web of Science dataset from the fields of Information Science and Library Science. The dataset contains keyword co-occurrence data and document-specific information, which is used to build the heterogeneous text graph for link prediction.

To use the dataset, please refer to the original dataset source. Alternatively, you can use the provided sample data for testing the model.
## Usage
1.webofsci File clean:
```bash
python 1_webofsci_File_clean.py
```
2.build graph:
```bash
python 2_build_graph.py
```
3.training (our method):
```bash
python 3_training.py
```
4.extract feature:
```bash
python 4_feature_extract.py
```
5.Comparative Experiment (traditional):
```bash
python 5_Comparative_Experiment_traditional.py
```
6.Comparative Experiment (GCN):
```bash
python 6_Comparative_Experiment_GCN.py
```
## Results
The proposed GCN-based approach achieved an AUC of 93.46% and an F1 score of 86.38%, outperforming traditional machine learning models such as XGBoost and SVM.

| **Method**                               | **Accuracy (%)** | **Precision (%)** | **Recall (%)** | **F1 (%)** | **AUC (%)** |
|------------------------------------------|------------------|-------------------|----------------|------------|-------------|
| Naive Bayes (TF-IDF)                     | 74.82            | 73.28             | 78.14          | 75.63      | 79.12       |
| Naive Bayes (TF-IDF+LDA)                 | 75.07            | 73.50             | 78.39          | 75.87      | 79.37       |
| Logistic Regression (TF-IDF)             | 72.22            | 75.47             | 65.85          | 70.33      | 79.30       |
| Logistic Regression (TF-IDF+LDA)         | 74.52            | 78.69             | 67.25          | 72.52      | 82.59       |
| Random Forest (TF-IDF)                   | 80.86            | 83.24             | 77.29          | 80.16      | 88.87       |
| Random Forest (TF-IDF+LDA)               | 81.20            | 83.50             | 77.77          | 80.53      | 89.12       |
| XGBoost (TF-IDF)                         | 76.14            | 73.00             | 82.99          | 77.67      | 84.40       |
| XGBoost (TF-IDF+LDA)                     | 79.93            | 78.59             | 82.27          | 80.39      | 88.83       |
| SVM (TF-IDF)                             | 70.38            | 74.04             | 62.76          | 67.94      | 76.34       |
| SVM (TF-IDF+LDA)                         | 72.46            | 77.12             | 63.86          | 69.87      | 78.92       |
| GCN                                      | 83.31            | 82.59             | 84.42          | 83.50      | 89.62       |
| **Our method** (${R_{w\sim w}}$ + ${R_{w\sim d}}$) | **84.90**         | **84.73**          | **86.86**       | **85.79**    | **92.14**    |
| **Our method** (${R_{w\sim w}}$ + ${R_{w\sim d}}$ + ${R_{w\sim t}}$) | **85.32**         | **84.86**          | **87.95**       | **86.38**    | **93.46**    |

**Table 1:** Comparison of the prediction performance of our method with a trivial GCN and traditional methods.
