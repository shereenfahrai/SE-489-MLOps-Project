# Fake vs Real News Detection

A machine learning project for detecting fake and real news articles.

## 1. Team Information
- [ ] Team Name: *The* Team
- [ ] Team Members (Name & Email): Shereen Fahrai (sfahrai@depaul.edu) and Liangcai Xie (lxie9@depaul.edu)
- [ ] Course & Section: SE 489 ML Engineering for Production (MLOps), Sections 910 and 930

## 2. Project Overview
- [ ] Brief summary of the project (2-3 sentences): This project builds a machine learning pipeline to detect fake versus real news articles. In order to ensure reproducibility and scalability, we use an LSTM‑based natural language processing model, integrated within an MLOps framework. 
- [ ] Problem statement and motivation: The spread of misinformation online can have very serious social, political, and economic consequences. To address this, automating the accurate detection of fake news can help platforms and end users filter unreliable content and make better informed decisions.  
- [ ] Main objectives:
    - Develop a text‑classification model to distinguish fake news from real news with high accuracy.  
    - Establish a reproducible, version‑controlled MLOps pipeline (data, experiment tracking, and deployment).  
    - Integrate third‑party tools (such as MLflow) for experiment tracking and model management.  
    - Evaluate the model using accuracy, precision, recall, and F1-score.
    - Containerize the training and prediction scripts using Docker to ensure environment consistency and portability.
    - Prepare for a user‑friendly GUI interface in later phases.


## 3. Project Architecture Diagram
![Architecture Diagram](reports/figures/ml_pipeline_architecture.png)

## 4. Phase Deliverables
- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions
- [ ] How to set up the environment (conda/pip, requirements.txt, Docker, etc.): 

    ### Prerequisites

    Before proceeding, make sure you have the following installed:
    - Python 3.11
    - Anaconda or Miniconda
    - Make
    - Docker

    ### Create and Activate Environment
    ```
        make create_environment
        make activate_environment
    ```
    ### Install Dependencies
    ```    
        make dev_requirements
    ```
- [ ] How to run the code and reproduce results
    ### Run Data Processing Pipeline
    ``` 
        make data
    ```
    ### Run Baseline Model
    ``` 
        make baseline_model
    ```
    ### Train Model (LSTM)
    ``` 
        make train_model
    ```
    ### Run Predictions
    ``` 
        make predict_model
    ```
    ### Check Code Quality
    ```
        make lint
        make typecheck
        make format
        make fix
    ```
    ### Run Docker Containers

    To containerize and run the ML pipeline in a clean, modular, and reproducible way, Docker images are provided for both training and prediction. A shared volume is used to persist the trained model and tokenizer so that they can be accessed across containers.

    #### Shared Volume Setup

    We mount a local `models/` folder to `/app/models` inside each container using Docker’s `-v` flag, hence allowing files saved during model training to be available for prediction without modifying or rebuilding images.

    #### Create Local `models/` Directory

    From the root of the project, run:
    ```
    mkdir models
    ```
    #### Build Docker Images
    ```
    docker build -f dockerfiles/train_model.dockerfile -t trainer .
    docker build -f dockerfiles/predict_model.dockerfile -t predictor .
    ```
    #### Run Training Container (Saves Trained Model to Volume)
    Windows Command Prompt (CMD):
    ```
    docker run -it --rm -v "%cd%/models:/app/models" trainer
    ```
    Windows PowerShell:
    ```
    docker run -it --rm -v ${PWD}/models:/app/models trainer
    ```
    Linux/macOS:
    ```
    docker run -it --rm -v $(pwd)/models:/app/models trainer
    ```
    This will run the training script and save lstm_model.h5 and tokenizer.pkl to shared models/ folder on host machine!
    #### Run Prediction Container (Loads Trained Model from Volume)
    Windows Command Prompt (CMD):
    ```
    docker run -it --rm -v "%cd%/models:/app/models" predictor
    ```
    Windows PowerShell:
    ```
    docker run -it --rm -v ${PWD}/models:/app/models predictor
    ```
    Linux/macOS:
    ```
    docker run -it --rm -v $(pwd)/models:/app/models predictor
    ```
    This loads trained model and tokenizer from shared volume and runs inference on test data!

## 6. Contribution Summary
- [ ] Briefly describe each team member's contributions:
    - Shereen Fahrai: 
        - Led data ingestion and preprocessing for Phase 1. 
        - Implemented `make_dataset.py` to automate merging, labeling, text cleaning, and stratified splitting of raw data. 
        - Set up the data pipeline outputs (`clean_data.csv`, `train.csv`, `predict.csv`). 
        - Maintained the `Makefile` with reproducible CLI commands. 
        - Contributed to documentation in `README.md` and `PHASE1.md`.
        - Led containerization for Phase 2. 
        - Created two separate Docker images for training and prediction, each with its own Dockerfile and volume mounting for model output persistence. 
        - Updated project documentation (`README.md`, `PHASE2.md`) with Docker usage instructions.
 
    - Liangcai Xie: 
        - Led model development and environment setup. 
        - Refactored the baseline LSTM into a PyTorch module, built `train_model.py` for training, logging, and model saving. 
        - Integrated MLflow for experiment tracking, and authored `visualize.py` for plotting training results
        - Created the architecture diagram. 
        - Contributed to documentation across `README.md` and `PHASE1.md`.


## 7. References
- [ ] List of datasets, frameworks, and major third-party tools used:
    - Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download
    - Baseline model code adapted from: https://www.kaggle.com/code/yossefmohammed/true-and-fake-news-lstm-accuracy-97-90/notebook
    - Tools: TensorFlow, Keras, NLTK, MLflow, Ruff, Mypy, PyTorch, Docker





## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── fake_news_detection  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
