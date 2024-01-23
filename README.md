Stage 1-2

How to run:
1. python3 -m venv venv
2. source ./venv/bin/activate
3. pip3 install -r requirements.txt
4. source .env
5. python3 ./train.py

Stage 3

How to run (assuming mlflow is installed):
1. source .env
2. mlflow run . --env-manager=local --experiment-name=kinopoisk

