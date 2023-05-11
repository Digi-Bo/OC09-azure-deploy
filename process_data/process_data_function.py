
#### Ce fichier est conçu sur la base développé dans hybrid_recommender.py
#### Il est adapté pour être exécuté dans un environnement Azure Functions

import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

import numpy as np
import pandas as pd
import glob
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from surprise import dump
import json

import io
import os

def load_data_from_blob_storage(container_name, blob_name, file_type):
    connection_string = os.environ["AzureWebJobsStorage"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    data = blob_client.download_blob()
    
    if file_type == 'csv':
        return pd.read_csv(io.BytesIO(data.readall()))
    elif file_type == 'pickle':
        return pickle.loads(data.readall())
    else:
        raise ValueError("Unsupported file_type")



container_name = "oc09dataglobo"

# Charger les données
articles_metadata = load_data_from_blob_storage(container_name, "articles_metadata.csv", "csv")
articles_metadata['datetime'] = pd.to_datetime(articles_metadata['created_at_ts'] / 1000, unit='s')

clicks = load_data_from_blob_storage(container_name, "clicks.csv", "csv")

articles = load_data_from_blob_storage(container_name, "articles_embeddings.pickle", "pickle")

# Charger le modèle de filtrage collaboratif
algo = load_data_from_blob_storage(container_name, "model.dump", "pickle")



def get_cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def recommend_articles(user_id, ref_start_date, ref_end_date, pred_start_date, pred_end_date):
    clicks['datetime'] = pd.to_datetime(clicks['click_timestamp'] / 1000, unit='s')
    user_click = clicks[clicks['user_id'] == user_id].sort_values('click_timestamp', ascending=False)

    # Filtrer les articles lus pendant la période de référence
    mask = (user_click['datetime'] > ref_start_date) & (user_click['datetime'] <= ref_end_date)
    ref_period = user_click.loc[mask]

    # Filtrer les articles lus pendant la période de prédiction
    mask = (user_click['datetime'] > pred_start_date) & (user_click['datetime'] <= pred_end_date)
    pred_period = user_click.loc[mask]

    last_article = ref_period['click_article_id'][:1].iloc[0]

    # Filtrer les articles publiés pendant la période de prédiction
    mask = (articles_metadata['datetime'] > pred_start_date) & (articles_metadata['datetime'] <= pred_end_date)
    pred_period_articles = articles_metadata.loc[mask]

    # Content-based Filtering
    cbf_scores = []
    for idx in pred_period_articles['article_id'].tolist():
        sim = get_cosine_similarity(np.array(articles[last_article]), np.array(articles[idx]))
        if sim < 0.25:
            cbf = 0
        elif sim < 0.50:
            cbf = 1
        elif sim < 0.75:
            cbf = 2
        else:
            cbf = 3
        cbf_scores.append([user_id, idx, sim, cbf])

    cbf_scores = pd.DataFrame(cbf_scores, columns=['user_id', 'article_id', 'sim', 'CBF'])

    # Collaborative Filtering
    cf_scores = []
    for article_id in pred_period_articles['article_id'].tolist():
        rating = algo.predict(user_id, article_id)
        cf_scores.append([rating.uid, rating.iid, rating.est, round(rating.est)])

    cf_scores = pd.DataFrame(cf_scores, columns=['user_id', 'article_id', 'raw', 'CF'])
    cf_scores = cf_scores.sort_values(by=['raw'], ascending=False)

    # Hybrid Recommender
    hybrid = pd.merge(cf_scores, cbf_scores, on=['user_id', 'article_id'])
    hybrid['score'] = hybrid['CF'] * 2 + hybrid['CBF']

    hybrid = hybrid.sort_values(by='score', ascending=False)
    top_articles = hybrid['article_id'][:5].tolist()

    return top_articles


def main(req: func.HttpRequest) -> func.HttpResponse:
    user_id = int(req.params.get("user_id"))
    ref_start_date = pd.to_datetime(req.params.get("ref_start_date"))
    ref_end_date = pd.to_datetime(req.params.get("ref_end_date"))
    pred_start_date = pd.to_datetime(req.params.get("pred_start_date"))
    pred_end_date = pd.to_datetime(req.params.get("pred_end_date"))

    top_articles = recommend_articles(user_id, ref_start_date, ref_end_date, pred_start_date, pred_end_date)

    return func.HttpResponse(json.dumps({"top_articles": top_articles}), status_code=200)
