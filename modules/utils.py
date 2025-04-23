import joblib
import torch
import numpy as np


def load_preprocessor(path='preprocessor.joblib'):
    """
    Загружает sklearn ColumnTransformer для предобработки признаков.
    """
    return joblib.load(path)


def load_model(model_path: str) -> float:
    """
    Загружает сохранённую модель и делает предсказание по входным данным.
    """
    return joblib.load(model_path)


def load_torch_model(model_class, input_dim, path='mlp_model.pth', **model_kwargs):
    """
    Загружает PyTorch модель из файла.

    Parameters:
        model_class: класс модели, например TorchMLPRegressor
        input_dim: размер входного вектора
        path: путь к сохранённым весам модели
        model_kwargs: дополнительные параметры для инициализации модели

    Returns:
        model: загруженная модель
    """
    model = model_class(input_dim=input_dim, **model_kwargs)
    model.load_model(path)
    return model


def preprocess_input(df, preprocessor):
    """
    Применяет предобработку к DataFrame и возвращает numpy-массив.

    Parameters:
        df: pandas DataFrame
        preprocessor: обученный ColumnTransformer

    Returns:
        X: numpy.ndarray
    """
    X_processed = preprocessor.transform(df)
    return X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
