import torch
import numpy as np
from modules.utils import load_model, load_torch_model, load_preprocessor
from modules.mlp import TorchMLPRegressor
from catboost import CatBoostRegressor, Pool
import joblib


rf_model_path = "models/random_forest_pipeline.pkl"
catboost_model_path = "models/catboost_model.cbm"
mlp_model_path = "models/mlp_model.pth"
preprocessor_path = "models/preprocessor_mlp.joblib"
all_scores_path = "models/all_scores.pkl"

# 1. Числовые признаки
numeric_features = [
    'num_published_lectures',
    'num_published_practice_tests', 'content_length_practice_test_questions',
    'content_hours', 'num_caption_locales', 'num_locales', 'num_instructors'
]

# 2. Категориальные признаки
categorical_features = [
    'locale', 'is_paid', 'has_closed_caption', 'instructional_level_simple'
]

# 3. Кластеры (мы оставляем их как числовые фичи напрямую)
cluster_features = [
    'description_embedding_cluster', 'instructor_embedding_cluster'
]

X_features = numeric_features + categorical_features + cluster_features


def get_random_forest_prediction(features):
    rf_pipeline = load_model(rf_model_path)
    return rf_pipeline.predict(features)[0]



def get_catboost_prediction(features):
    features = features[X_features]
    
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(catboost_model_path)
    cat_features = np.where([True if col in categorical_features else False for col in features.columns])[0].tolist()

    X_pool = Pool(data=features, cat_features=cat_features)
    return catboost_model.predict(X_pool)[0]
    


def get_mlp_prediction(features):
    preprocessor_mlp = load_preprocessor(preprocessor_path)

    X_processed = preprocessor_mlp.transform(features)
    X_tensor = torch.tensor(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed, dtype=torch.float32)
    
    model = load_torch_model(TorchMLPRegressor, X_tensor.shape[1], mlp_model_path)
    with torch.no_grad():
        prediction = model.predict(X_tensor)[0]
    return prediction


def get_ensemble_prediction(features):
    rf_pred = get_random_forest_prediction(features)
    cat_pred = get_catboost_prediction(features)
    mlp_pred = get_mlp_prediction(features)
    
    return (mlp_pred + rf_pred + cat_pred) / 3



def success_category(score):
    if score < 20:
        return "Низкий"
    elif score < 30:
        return "Средний"
    elif score < 40:
        return "Высокий"
    else:
        return "Отличный"
    
    
def percentile_rank(score):
    all_scores = joblib.load(all_scores_path)
    return np.mean(all_scores <= score) * 100