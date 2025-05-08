# PHASE 1: Project Design & Model Development

## 1. Project Proposal
- [ ] **1.1 Project Scope and Objectives**
  - [ ] Problem statement: The spread of online misinformation poses serious social, political, and economic risks. Since manual fact‑checking at scale is infeasible, an automated system is necessary to identify and filter fake news articles.
  - [ ] Project objectives and expected impact:
    1. Build a text‑classification model that flags news articles as fake (0) or real (1) with ≥ 95 % accuracy.  
    2. Develop a reproducible, version‑controlled pipeline that ingests raw data, preprocesses text, trains the model, and evaluates performance.  
    3. Integrate experiment tracking via Weights & Biases to log hyperparameters and metrics.  
    4. Lay the groundwork for a simple GUI in Phase 3, allowing users to input articles and receive real-time classifications.
  - [ ] Success metrics (targets):
    *These are the performance goals we hope to reach on the hold‑out set; our baseline from the adapted notebook already achieved ~97.9 % accuracy!*
    - Accuracy ≥ 95 %  
    - Precision ≥ 90 %  
    - Recall ≥ 90 %  
    - F1‑score ≥ 0.90  

  - [ ] 300+ word project description: 
  
The goal of this project is to address the issue of misinformation by constructing a machine learning model that is able to categorize news articles as being either “fake” or “real.” Given the prevalence of misleading and completely falsified content circulating online, manual fact‑checking is insufficient to keep up with the volume of daily publications. Our overall objective is to produce an end-to-end reproducible pipeline— from raw data input to preprocessing, model training, testing, and simple deployment scaffolding— in a manner that may be iterated, monitored, and scaled in production.

We have thus far built our workflow from a Cookiecutter MLOps template, with a standardized repository setup and well-isolated modules for data, models, scripts, and tests. Data preprocessing is automated in a single Python script (make_dataset.py) that merges and labels the Kaggle “Fake and Real News” dataset, drops irrelevant columns, lowercases and tokenizes text, lemmatizes with NLTK, removes stopwords, and outputs a cleaned CSV. All steps are version-controlled, with development requirements using Conda and pip, and adherence to code quality by Ruff and Mypy.

To build our model, we will utilize TensorFlow and Keras to design a Long Short‑Term Memory (LSTM) neural network. LSTMs are selected due to their sensitivity to long‑range dependencies and fine‑grained linguistic cues at the paragraph level- important for separating misleading language patterns from true reporting. We will include dropout and global max‑pooling layers to avoid the risks of overfitting, and we’ll track hyperparameters, metrics, and artifacts with Weights & Biases; we have chosen to use this third‑party tool because it is integrated well with Keras using a WandbCallback, supplying us with real‑time dashboards and experiment versioning.

Our base dataset is in the form of two CSV files, Fake.csv (label 0) and True.csv (label 1), with more than 44,000 articles. This pre‑labeled, balanced dataset supports quick prototyping without relabeling overhead and without data augmentation. We will assess our model with accuracy, precision, recall, and F1‑score on a held‑out test set, aiming for ≥ 95 % accuracy and good precision/recall tradeoffs.

By the conclusion of Phase 1, we will have a clean, well-documented data pipeline and a baseline LSTM classifier whose performance is tracked in Weights & Biases. Future phases will build on this using CI/CD, model drift monitoring, and ultimately, a simple GUI user interface for inputting URLs or text and getting real‑time predictions.

- [ ] **1.2 Selection of Data**
  - [ ] Dataset(s) chosen and justification:  
    - “fake-and-real-news-dataset” (by Clément Bisaillon) on Kaggle includes over 44,000 articles, evenly balanced between fake and real labels, which removes the need to manually correct class imbalance. Moreover, it has a straightforward CSV format and clear labeling schema, which allows for quick ingestion into our preprocessing pipeline, hence reducing setup time and enabling quick iteration.
  - [ ] Data source(s) and access method:
    - Downloaded zip file of dataset on Kaggle (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download)
  - [ ] Preprocessing steps:
    1. Merged `Fake.csv` (label 0) and `True.csv` (label 1).  
    2. Dropped unused columns: `title`, `subject`, `date`.  
    3. Lowercased, removed non‑alphabetic characters and extra whitespace.  
    4. Tokenized text with NLTK’s `punkt` tokenizer.  
    5. Lemmatized tokens via WordNet.  
    6. Removed English stopwords and tokens shorter than 4 characters.  
    7. Rejoined tokens into cleaned sentences and saved to `data/processed/clean_data.csv`.
    8. Split cleaned data into:
      - `train.csv` (90 %, stratified) for model training  
      - `predict.csv` (10 %, stratified) for inference/testing
- [ ] **1.3 Model Considerations**
  - [ ] Model architecture(s) considered
  - [ ] Rationale for model choice:
     - Kaggle’s baseline notebook that we used implemented an LSTM and achieved 97.90 % test accuracy, proving its effectiveness on this dataset.
     - LSTMs capture long‑range dependencies in text, which is crucial for spotting deceptive language across sentences.
  - [ ] Source/citation for any pre-built models:
    - Baseline model code adapted from: https://www.kaggle.com/code/yossefmohammed/true-and-fake-news-lstm-accuracy-97-90/notebook

- [ ] **1.4 Open-source Tools**
  - [ ] Third-party package(s) selected (not PyTorch or course-used tools)
    - Weights & Biases: for logging hyperparameters, metrics, and artifacts with minimal code changes
    - NLTK: used to standardize cleaning, as it provides text-processing utilities (e.g. `punkt`, `punkt_tab`, `stopwords`, `wordnet`)
  - [ ] Brief description of how/why used


## 2. Code Organization & Setup
- [ ] **2.1 Repository Setup**
  - [ ] GitHub repo created
  - [ ] Cookiecutter or similar structure used
- [ ] **2.2 Environment Setup**
  - [ ] Python virtual environment
  - [ ] requirements.txt or environment.yml
  - [ ] (Optional) Google Colab setup

## 3. Version Control & Collaboration
- [ ] **3.1 Git Usage**
  - [ ] Regular commits with clear messages
  - [ ] Branching and pull requests
- [ ] **3.2 Team Collaboration**
  - [ ] Roles assigned
  - [ ] Code reviews and merge conflict resolution

## 4. Data Handling
- [ ] **4.1 Data Preparation**
  - [ ] Cleaning, normalization, augmentation scripts
- [ ] **4.2 Data Documentation**
  - [ ] Description of data prep process

## 5. Model Training
- [ ] **5.1 Training Infrastructure**
  - [ ] Training environment setup (e.g., Colab, GPU)
    We trained the model locally using a Conda-managed Python 3.11 environment on a CPU-only machine. All dependencies are defined in `requirements.txt` and managed via a `Makefile` to ensure consistent reproducibility.

    Although no external GPU or Colab was used in this phase, the codebase is fully portable. It can be run on Colab or GPU-accelerated machines simply by setting up the same environment and installing the required dependencies.

Experiment tracking was handled via **MLflow**, and model artifacts were saved for future deployment or evaluation.

- [ ] **5.2 Initial Training & Evaluation**
  - [ ] Baseline model results
  A baseline model using **TF-IDF + Logistic Regression** was implemented to set a performance benchmark. It achieved the following on the test set:
    - **Accuracy**: 98.29%  
    - **Precision**: 97.96%  
    - **Recall**: 98.52%
    - **F1 Score**: 0.9824 
  - [ ] Evaluation metrics
      For the LSTM model, we evaluated the following metrics:
    - **Accuracy**: 98.61%  
    - **Loss**: 0.04  
    - **Precision**: 98.79%
    - **Recall**: 98.30%
    - **F1 Score**: 0.9855
    
    ####  Prediction Performance
    After training, we ran inference on a separate holdout set `predict.csv` (10% of original dataset). This set was not used during training or validation.

    The model achieved the following metrics on this unseen data:
    - **Accuracy**: 98.62%  
    - **Precision**: 98.74%  
    - **Recall**: 98.41%
    - **F1 Score**: 0.9857

    A confusion matrix for this prediction run is saved as `predict_confusion_matrix.png` under `reports/figures/`.

    This confirms that the model generalizes well and maintains performance on real-world test samples.


    These were logged automatically with **MLflow**, and confusion matrix plots were saved using the `plot_confusion_matrix()` function.

    The performance indicates that the LSTM model significantly outperforms the baseline in terms of both recall and F1 score, suggesting better generalization to unseen data.

## 6. Documentation & Reporting
- [ ] **6.1 Project README**
  - [ ] Overview, setup, replication steps, dependencies, team contributions
- [ ] **6.2 Code Documentation**
  - [ ] Docstrings, inline comments, code style (ruff), type checking (mypy), Makefile docs
