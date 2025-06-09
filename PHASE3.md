# PHASE 3: Continuous Machine Learning (CML) & Deployment

## 1. Continuous Integration & Testing
- [ ] **1.1 Unit Testing with pytest**
  - [ ] Test scripts for data processing, model training, and evaluation
  - `make_dataset.py`: Validates raw CSV loading, label assignment, text cleaning, and proper train/test split
  - `model.py`: Confirms model layer structure, output shapes, and successful compilation with correct metrics
  - `predict_model.py`: Tests inference pipeline, prediction consistency, and confusion matrix generation

  - [ ] Documentation of the testing process and example test cases
  - All test files are located in the `tests/` directory and follow `test_*.py` naming convention:
  - `tests/test_make_dataset.py`
  - `tests/test_train_model.py`
  - `tests/test_predict_model.py`

  - Each file includes:
      - Clear separation of unit test functions
      - Explanatory docstrings and inline comments
      - Use of lightweight dummy data for isolated logic validation
    - Example test command:
      ```bash
      pytest tests/ --maxfail=3 --disable-warnings --tb=short
      ```


- [ ] **1.2 GitHub Actions Workflows**
  - [ ] CI workflows for running tests, DVC, code checks (e.g., ruff), Docker builds
  - Created four GitHub Actions workflows: `ci.yml`, `cml-eval.yml`, `deploy.yml`, and `train-vertexai.yml`.

  - [ ] Workflow YAML files included
  - `ci.yml` is triggered on every push or pull request to the `main` branch. It includes:
      - Code formatting checks using `ruff` and `black`
      - Type checking using `mypy`
      - Running all unit tests in `tests/` using `pytest`

    - `cml-eval.yml` triggers on pull requests. It:
      - Runs model evaluation (`predict_model.py`)
      - Generates a confusion matrix figure
      - Uses CML to automatically comment the report (with the figure) on the PR
      - Uses a Personal Access Token (`CML_PAT`) as GitHub secret

    - `deploy.yml` handles continuous deployment:
      - Builds and pushes Docker image to GCP Artifact Registry
      - Downloads model weights (if needed) from GitHub Releases
      - Deploys to Google Cloud Run automatically on `main` merge

    - `train-vertexai.yml` runs training jobs on GCP Vertex AI:
      - Submits a training pipeline using `gcloud`
      - Logs training output
      - Artifacts and metrics optionally stored in GCS or Vertex AI Experiments

  - These workflows enable full CI/CD + CML pipelines:
    - Code checked and tested on push
    - Evaluation results visualized on PR
    - Deployments automated on merge
    - Cloud training jobs triggered manually or on schedule

  - All workflow YAMLs are located in `.github/workflows/` and can be triggered manually or via PR/push.

- [ ] **1.3 Pre-commit Hooks**
  - [ ] Pre-commit config and setup instructions
  -  -  Configured `.pre-commit-config.yaml` with the following hooks:
    - `trailing-whitespace`
    - `end-of-file-fixer`
    - `check-yaml`
    - `black`
    - `ruff`
    - `mypy`
  -  Installed pre-commit locally and verified the hooks trigger on each commit to ensure consistent code quality.
  -  ```pre-commit run --all-files```
  - These hooks ensure consistent style and catch errors before code reaches CI.


## 2. Continuous Docker Building & CML
- [ ] **2.1 Docker Image Automation**
  - [ ] Automated Docker builds and pushes (GitHub Actions)
  - Wrote a custom `fastapi_app.dockerfile` that:
  - Uses `python:3.11-slim` as the base image
  - Installs dependencies from `requirements.txt`
  - Copies FastAPI application code
  - Downloads `lstm_model.h5` and `tokenizer.pkl` from the GitHub release using `wget` into the `/app/models` directory
  - [ ] Dockerfile and build/push instructions for Docker Hub and GCP Artifact Registry
  - Automated Docker image builds and pushes to:
    - `us-central1-docker.pkg.dev/fake-news-api-project/fastapi-repo/fastapi-api`
  -  This process is fully integrated into the `deploy.yml` workflow and triggered by commits to the `main` branch.

- [ ] **2.2 Continuous Machine Learning (CML)**
  - [ ] CML integration for automated model training on PRs
  We integrated [CML (Continuous Machine Learning)](https://cml.dev/) into our GitHub Actions workflow to automate model evaluation every time a pull request is opened or updated on the `main` branch. The workflow performs the following:

  - Checks out the PR branch
  - Installs Python dependencies
  - Runs the model evaluation script (`predict_model.py`)
  - Generates a confusion matrix figure
  - Posts the evaluation results as a PR comment using `cml comment create`

  We configured authentication using a **Personal Access Token (PAT)** with the required permissions (`repo`, `write:discussion`, `write:packages`, `read:org`), stored as a GitHub Actions secret named `CML_PAT`.


  - [ ] Example CML outputs (metrics, visualizations)
  - On each PR, CML automatically generates and posts a **confusion matrix** image like the one below, which gives visual feedback on prediction quality.
    Sample comment (automatically posted by CML):
  ![Confusion Matrix](reports/figures/CML_PR_COMMENT.png)

  - [ ] Setup and usage documentation

  To reproduce or extend our setup:

  1. **Workflow file:** `.github/workflows/cml-eval.yml`
  2. **Dependencies:** Listed in `requirements.txt`
  3. **Evaluation script:** `fake_news_detection/predict_model.py`
  4. **Token setup:**
     - Generate a GitHub **Classic PAT** with the following scopes:
       `repo`, `write:discussion`, `write:packages`, `read:org`
     - Save the token in repo secrets:
       `Settings → Secrets and variables → Actions → New repository secret`
       Name: `CML_PAT`, Value: your generated token
  5. **Trigger:** Any pull request on the `main` branch will trigger the evaluation workflow.

## 3. Deployment on Google Cloud Platform (GCP)
- [ ] **3.1 GCP Artifact Registry**
  - [ ] Steps for creating and pushing Docker images to GCP
  -  Created and configured Artifact Registry in the `fake-news-api-project` on GCP.
  -  Used `google-github-actions/setup-gcloud` in GitHub Actions with a service account stored in `GCP_SA_KEY` secret.
  -  Successfully pushed Docker image to `fastapi-repo` in `us-central1`.
- [ ] **3.2 Custom Training Job on GCP**
  - [ ] Vertex AI/Compute Engine job setup and documentation
  -  Implemented training logic in `train_model.py` and containerized it using `dockerfiles/train_model.dockerfile`.
  -  Built and pushed the training image to Artifact Registry:
     ```bash
     us-central1-docker.pkg.dev/fake-news-api-project/trainer-repo/train-job
     ```
     -  Submitted the training job via GitHub Actions using `gcloud ai custom-jobs create`:
    ```bash
        gcloud ai custom-jobs create \
          --region=us-central1 \
          --display-name=vertex-lstm-train \
          --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/fake-news-api-project/trainer-repo/train-job
    ```
    Job submission is automated via .github/workflows/train-vertexai.yml, which supports manual triggering through workflow_dispatch.
      ![Vertex](reports/figures/VertexAI.png)
      ![Vertex](reports/figures/VertexAI_GCP.png)
  - [ ] Data storage in GCP bucket
  - Training data train.csv is stored in GCS at:
   https://storage.googleapis.com/mlops_fake_news/clean.csv
  - Model artifacts (lstm_model.h5, tokenizer.pkl) are saved back to the same bucket using gcsfs:
  ```python
    fs.put("/tmp/lstm_model.h5", "gs://mlops_fake_news/lstm_model.h5")
    fs.put("/tmp/tokenizer.pkl", "gs://mlops_fake_news/tokenizer.pkl")
  ```
  Public read access is enabled on the bucket to support model downloading from Cloud Run.
  ![Data](reports/figures/data_store.png)


- [ ] **3.3 Deploying API with FastAPI & GCP Cloud Functions**
  - [ ] FastAPI app for model predictions
    - Created `/predict` POST endpoint in `main.py` using FastAPI.
    - Loads `lstm_model.h5` and `tokenizer.pkl` from `models/` directory.
    - Returns prediction result as JSON.
  - [ ] Deployment steps and API testing instructions
    - Dockerized app with `Dockerfile` using `uvicorn` server.
    - Built and tested image locally.
    - Pushed Docker image to GCP Artifact Registry.
    - Deployed container to GCP Cloud Run:
      ```bash
      gcloud run deploy fastapi-service \
        --image us-central1-docker.pkg.dev/[PROJECT_ID]/fastapi-repo/fastapi-api \
        --platform managed --region us-central1 --allow-unauthenticated
      ```
    - Verified API with `curl` by sending POST requests to `/predict`.

- [ ] **3.4 Dockerize & Deploy Model with GCP Cloud Run**
  - [ ] Containerization and deployment steps
    - Wrote `Dockerfile` using `python:3.11-slim` as base image.
    - Installed dependencies from `requirements.txt`.
    - Copied `fake_news_detection/` and `models/` (includes `lstm_model.h5` and `tokenizer.pkl`).
    - Exposed port `8080`, launched app using `uvicorn`.
    - Created GitHub Actions workflow `deploy-to-cloudrun.yml`:
      - Builds Docker image.
      - Pushes to Artifact Registry: `us-central1-docker.pkg.dev/fake-news-api-project/fastapi-repo/fastapi-api`.
      - Deploys to Cloud Run using `gcloud run deploy`.
  - [ ] Testing and result documentation
  - Verified deployment via the public Cloud Run endpoint.
    - Sent POST request to `/predict` with example input:
      ```bash
      curl -X POST https://fastapi-service-494247760906.us-central1.run.app/predict \
       -H "Content-Type: application/json" \
       -d '{"text": "This news is absolutely fake!"}'
      ```
    - Received valid prediction response `{ "prediction": 0.9986182451248169 }`.
    - Screenshots of successful deployment and API test included in documentation.
- [ ] **3.5 Interactive UI Deployment**
  - [ ] Streamlit or Gradio app for model demonstration
    - Implemented a clean and functional interactive web interface using **Gradio 4.27.0** to demonstrate our fake news classification model.
    - The app accepts raw article body text as input and returns a classification: "Fake" or "Real", along with a confidence percentage.
    - Input text is preprocessed using the same standardized tokenization and cleaning pipeline applied during model training, ensuring alignment between training and inference stages.
    - The Gradio app loads the LSTM model and tokenizer from the public Google Cloud Storage (GCS) bucket at runtime to ensure portability and consistency across deployments.
  - [ ] Deployment on Hugging Face platform
    - Deployed the Gradio app to **Hugging Face Spaces** under the name `sfahrai/fake-new-detection-ui`.
    - Model and tokenizer are fetched at runtime from:
      - `https://storage.googleapis.com/mlops_fake_news/lstm_model.h5`
      - `https://storage.googleapis.com/mlops_fake_news/tokenizer.pkl`
    - Hugging Face Space is configured to run the app using `gradio_app.py`, renamed as `app.py` during GitHub Actions deployment.
    - This setup allows the UI to remain lightweight and cloud-compatible, while ensuring full functionality in a public environment.
  - [ ] Integration of UI deployment into GitHub Actions workflow
    - Created `deploy_to_hf.yml` workflow inside `.github/workflows/` to automate Hugging Face deployment on every push to `main`.
    - The workflow:
      - Clones the Hugging Face Space repository using the `HF_TOKEN` GitHub secret
      - Copies the application interface code and all required files into the Hugging Face Space directory to support deployment and execution of the Gradio UI
      - Commits and pushes changes directly to the Hugging Face Space repo, triggering a UI rebuild
    - This setup ensures any updates to the UI are continuously deployed without manual intervention.
    - The deploy step in the GitHub Actions log confirms a successful push to the Hugging Face Space repository, which automatically triggers the UI rebuild upon push to main. A snapshot of this pipeline stage is shown below.

      ![UI Deployment Workflow](reports/figures/ui_pipeline_view.png)
  - [ ] Screenshots and usage examples
    - Below is an example of the interface in action:

      **Before Prediction:**

      ![UI Before Prediction](reports/figures/ui_before_prediction.png)

      **After Fake News Submission (AI-Generated):**

      ![UI After Prediction of Fake News](reports/figures/ui_fake_news_prediction.png)

      **After Real News Submission:**

      Source: https://www.nbcnews.com/news/us-news/kilmar-abrego-garcia-questions-rcna211601
      ![UI After Prediction of Real News](reports/figures/ui_real_news_prediction.png)

## 4. Documentation & Repository Updates
- [ ] **4.1 Comprehensive README**
  - [ ] Setup, usage, and documentation for all CI/CD, CML, and deployment steps
  - [ ] Screenshots and results of deployments
- [ ] **4.2 Resource Cleanup Reminder**
  - [ ] Checklist for removing GCP resources to avoid charges

---

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables. Focus on automation, deployment, and clear, reproducible instructions for all steps.
