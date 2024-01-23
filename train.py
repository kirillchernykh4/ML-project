import boto3
import io
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pymorphy2
from string import punctuation
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, hp, anneal, Trials

s3 = boto3.resource('s3')
s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


bucket = "datasets"
key = "kinopoisk_train.csv"
s3 = boto3.client(
    service_name='s3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=s3_endpoint_url,
    verify=False
)

print("loading dataset")

obj = s3.get_object(Bucket=bucket, Key=key)
data = obj['Body'].read().decode('utf-8')
df = pd.read_csv(io.StringIO(data))
df = df.replace(['neg', 'neu', 'pos'], [0, 1, 2])
df = df.drop(df[df.duplicated()].index)
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

def stop_finder(text):
    stop_list = ['*',"``","'","--","№",'«','»', '.','–','…','-', '“', '”' ]
    for stop in stop_list:
        if text.find(stop) == -1:
            pass
        else:
            text = ' '
    return text
    
#Обработка текста - убираем стоп-слова и пунктуацию, проводим к нормальной форме слова
def preprocessor(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in russian_stopwords\
              and stop_finder(token) != ' ' \
              and token.strip() not in punctuation]
    norm_tokens = []
    for token in tokens:
        norm_tokens.append(morph.parse(token)[0].normal_form)
        
    return norm_tokens


df['Tokens'] = df['review'].apply(preprocessor)
df.head()
df['Tokens'] = df['Tokens'].apply(lambda x: ' '.join(x))
df.head()

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(df['Tokens'], df['sentiment'], 
                                                    test_size=0.20, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_test)

logreg.fit(X_train_vec, y_train)
y_pred = logreg.predict(X_val_vec)
acc = round(accuracy_score(y_test, y_pred),4)
print("Accuracy score: ", acc)

space = {
    'max_features': hp.choice('max_features', [None, 500, 1000, 5000]),
    'C': hp.loguniform('C', -4, 2)
}

def objective(params):
    vectorizer = TfidfVectorizer(max_features=params['max_features'])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_test)
    
    clf = LogisticRegression(C=params['C'])
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_val_vec)
    return 1 - accuracy_score(y_test, y_pred)

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

with mlflow.start_run() as run:
    max_iter = 50

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    
    mlflow.log_param("model_type", "sklearn.linear_model.LogisticRegression")
    mlflow.log_param("vectorizer", "sklearn.feature_extraction.text.count_vectorizer")
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("train_set_size", y_train.shape[0])
    mlflow.log_param("test_set_size", y_test.shape[0])
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, model.predict(X_train_vec)))
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    log_model(model, "model", registered_model_name="review-sentiment")


print("done")