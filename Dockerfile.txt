FROM ghcr.io/mlflow/mlflow:v2.9.2

ADD requirements.txt .
RUN  --mount=type=cache,target=/root/.cache pip3 install -r requirements.txt
ADD train.py /

ENTRYPOINT [ "/usr/local/bin/python3" ]