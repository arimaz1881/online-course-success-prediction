import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
from typing import List
import joblib

kmeans_description = joblib.load('models/kmeans_description.pkl')
kmeans_instructor = joblib.load('models/kmeans_instructor.pkl')
bert_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_bert_embeddings(texts: List[str]) -> np.ndarray:
    """
    Получение эмбеддингов из текстов с помощью sentence-transformers.
    """
    
    text = '|'.join(texts)
    if not text:
        text = ""
    return bert_model.encode(text, show_progress_bar=False)

def prepare_features(df: pd.DataFrame, numeric_features: List[str], 
                     categorical_features: List[str], cluster_features: List[str]) -> ColumnTransformer:
    """
    Создание препроцессора на основе числовых, категориальных и кластерных признаков.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features + cluster_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor


def prepare_input_features(df: pd.DataFrame):
    
    level_map = {
        'All Levels': 0,
        'Beginner': 1,
        'Intermediate': 2,
        'Expert': 3
    }
    df['num_locales'] = 1
    df['num_caption_locales'] = df['caption_locales'].apply(len)
    df['num_instructors'] = df['instructors'].apply(len)
    df['instructional_level_simple'] = df['instructional_level_simple'].map(level_map)
    
    desc_emb = df['description'].apply(extract_bert_embeddings)[0]
    inst_emb = df['instructors'].apply(extract_bert_embeddings)[0]
    
    df['description_embedding_cluster'] = kmeans_description.predict([desc_emb])[0]
    df['instructor_embedding_cluster'] = kmeans_instructor.predict([inst_emb])[0]
    
    
    return df
