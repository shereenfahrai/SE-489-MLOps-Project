# PHASE 1: Project Design & Model Development

## 1. Project Proposal
- [ ] **1.1 Project Scope and Objectives**
  - [ ] Problem statement: The spread of online misinformation poses serious social, political, and economic risks. Since manual fact‑checking at scale is infeasible, an automated system is necessary to identify and filter fake news articles.
  - [ ] Project objectives and expected impact:
    1. Build a text‑classification model that flags news articles as fake (0) or real (1) with ≥ 95 % accuracy.  
    2. Develop a reproducible, version‑controlled pipeline that ingests raw data, preprocesses text, trains the model, and evaluates performance.  
    3. Integrate experiment tracking via MLflow to log hyperparameters and metrics.  
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

To build our model, we will utilize TensorFlow and Keras to design a Long Short‑Term Memory (LSTM) neural network. LSTMs are selected due to their sensitivity to long‑range dependencies and fine‑grained linguistic cues at the paragraph level- important for separating misleading language patterns from true reporting. We will include dropout and global max‑pooling layers to reduce overfitting, and we’ll track hyperparameters, metrics, and artifacts using MLflow, a third‑party tool that enables experiment logging, result comparison, and artifact versioning across training runs. We chose MLflow due to its flexibility, ease of integration with PyTorch, and ability to work well in local development environments without additional configuration.

Our base dataset is in the form of two CSV files, Fake.csv (label 0) and True.csv (label 1), with more than 44,000 articles. This pre‑labeled, balanced dataset supports quick prototyping without relabeling overhead and without data augmentation. We will assess our model with accuracy, precision, recall, and F1‑score on a held‑out test set, aiming for ≥ 95 % accuracy and good precision/recall tradeoffs.

By the conclusion of Phase 1, we will have a clean, well-documented data pipeline and a baseline LSTM classifier whose performance is tracked in MLflow. Future phases will build on this using CI/CD, model drift monitoring, and ultimately, a simple GUI user interface for inputting URLs or text and getting real‑time predictions.

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
  - [ ] Brief description of how/why used
    - MLflow: used to track experiments, metrics, and artifacts for each model training run
    - NLTK: used to standardize cleaning, as it provides text-processing utilities (e.g. `punkt`, `punkt_tab`, `stopwords`, `wordnet`)

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
  - Data preparation is handled in `make_dataset.py`, which automates the following:
    - Loads and merges the raw `True.csv` and `Fake.csv` files
    - Assigns binary labels (1 = real, 0 = fake)
    - Drops unnecessary columns (`title`, `subject`, `date`)
    - Applies text cleaning: lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization using NLTK
    - Outputs the cleaned dataset to `clean_data.csv`
    - Splits the cleaned data into `train.csv` (90%) and `predict.csv` (10%) using stratified sampling to preserve the original label distribution
  - No data augmentation was used, as the dataset was already balanced.
- [ ] **4.2 Data Documentation**
  - The data preprocessing and splitting steps are fully documented in:
    - `README.md` : Setup instructions and data folder structure
    - `PHASE1.md` : Section 1.2 includes the dataset source, justification, and all preprocessing steps


## 5. Model Training
- [ ] **5.1 Training Infrastructure**
  - [ ] Training environment setup (e.g., Colab, GPU)
    We trained the model locally using a Conda-managed Python 3.11 environment on a CPU-only machine. All dependencies are defined in `requirements.txt` and managed via a `Makefile` to ensure consistent reproducibility.

    Although no external GPU or Colab was used in this phase, the codebase is fully portable. It can be run on Colab or GPU-accelerated machines simply by setting up the same environment and installing the required dependencies.

Experiment tracking was handled via **MLflow**, and model artifacts were saved for future deployment or evaluation.

- [ ] **5.2 Initial Training & Evaluation**

    To establish baseline performance, we first trained a Logistic Regression model using TF-IDF features. We applied 5-fold Stratified Cross-Validation to evaluate its performance across folds and averaged the resulting metrics (see baseline_model.py). This provided a strong reference point for initial model quality.

    Building on this baseline, we then designed and trained a more complex LSTM model using TensorFlow/Keras. While the LSTM model did not apply cross-validation due to its computational cost, we split the training set into training and validation (80/20) during training via the `validation_data` parameter in Keras. This enabled us to monitor model generalization throughout training epochs.

    All training configurations—such as optimizer, dropout, learning rate, and architecture depth—were explicitly logged via MLflow, and model artifacts were saved for later reuse. This ensured that our results were well-documented, reproducible, and traceable for future improvement.
    
    
- [ ] Baseline model results
      A baseline model using **TF-IDF + Logistic Regression** was implemented to establish a performance benchmark.  
    To ensure robust evaluation, we used **5-fold Stratified Cross-Validation**, which preserves class distribution in each fold.  
    For each fold, the model was trained on 80% of the data and validated on the remaining 20%.  
    We computed **accuracy**, **precision**, **recall**, and **F1 score** for each fold, then averaged the results across folds to obtain a stable performance estimate.

  It achieved the following on the test set:
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

    A confusion matrix for this prediction run is saved as `predict_confusion_matrix.png` under `fake_news_detection/reports/figures/`.

    This confirms that the model generalizes well and maintains performance on real-world test samples.

    These were logged automatically with **MLflow**, and confusion matrix plots were saved using the `plot_confusion_matrix()` function.

    The performance indicates that the LSTM model significantly outperforms the baseline in terms of both recall and F1 score, suggesting better generalization to unseen data.
    
    #### Model Training Strategy and Hyperparameter Settings

    For both baseline and LSTM models, we split the data into 90% training and 10% inference (predict-only) using stratified sampling to preserve label distribution. The LSTM model was trained on this training set, with 20% further held out as validation during training.

    These settings were initially chosen based on values used in the adapted Kaggle baseline and then retained after observing high validation performance with no major signs of overfitting. MLflow was used to log all parameters and results for reproducibility and later tuning.

    #### Hyperparameter Tuning and Optimization (Planned)
    We did not yet perform extensive hyperparameter tuning (e.g., grid search or random search), but our code structure supports easy integration of such approaches using libraries like `optuna` or `scikit-learn GridSearchCV`.

    For now, we manually experimented with dropout and learning rate. Our results suggest that the current settings yield near-optimal generalization for this dataset.

    Ensemble methods (e.g., combining LSTM with TF-IDF + Logistic Regression, or voting classifiers) were considered but not yet implemented, as the current LSTM already exceeds the target metrics. However, future extensions could explore:
        - LSTM + CNN hybrid architectures
        - BERT-based fine-tuned transformer models
        - Ensemble voting of classical + neural models

    All training runs were versioned and tracked using MLflow, with model weights and tokenizer objects saved to disk. This ensures that future experiments (e.g., tuning or ensembling) can reproduce and extend from our current baseline.
        
        
- [ ] **5.3 Experiment Tracking and Artifact Logging** 
    We used **MLflow** to track all training experiments in this project. MLflow provides a centralized way to log:

    - Model hyperparameters:  
      `epochs`, `embedding_dim`, `lstm_units`, `dropout`, `learning_rate`, and more.

    - Evaluation metrics:  
      `test_accuracy`, `test_loss`, `test_f1_score`, `precision`.

    - Artifacts:  
      - Trained model: `lstm_model.h5`  
      - Tokenizer: `tokenizer.pkl`  
      - Confusion matrix: `train_confusion_matrix.png`, `predict_confusion_matrix.png`  
      - Accuracy/loss plots: `accuracy.png`, `loss.png`

    All training runs are stored locally under the `mlruns/` directory. You can launch the experiment UI dashboard using:

    ```bash
    mlflow ui --port 5000
    ```
    Then visit http://localhost:5000 in your browser to view and compare runs.

    Below is a snapshot of our MLflow UI showing a successful run:
    ![MLflow UI Screenshot](reports/figures/mlflow_ui.png)


## 6. Documentation & Reporting
- [ ] **6.1 Project README**
  - [ ] Overview, setup, replication steps, dependencies, team contributions
- [ ] **6.2 Code Documentation**
  - [ ] Docstrings, inline comments, code style (ruff), type checking (mypy), Makefile docs
