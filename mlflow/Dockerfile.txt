FROM ghcr.io/mlflow/mlflow:v2.9.2
# Install python packages
ADD requirements.txt .
RUN pip install -r requirements.txt