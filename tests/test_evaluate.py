# tests/test_evaluate.py
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.evaluate.evaluate import evaluate_prophet, evaluate_sarimax


@pytest.fixture
def mock_prophet_model():
    """Prophet 모델 Mock"""
    model = MagicMock()
    forecast_data = pd.DataFrame({
        'yhat': [15.0, 16.0, 17.0, 18.0, 19.0],
        'yhat_lower': [14.0, 15.0, 16.0, 17.0, 18.0],
        'yhat_upper': [16.0, 17.0, 18.0, 19.0, 20.0]
    })
    model.predict.return_value = forecast_data
    return model


@pytest.fixture
def test_data():
    """테스트용 데이터"""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='H')
    temps = [15.1, 15.9, 17.2, 17.8, 19.1]
    return pd.DataFrame({'time': dates, 'temp': temps})


@pytest.fixture
def temp_test_csv(test_data):
    """임시 테스트 CSV 파일"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


def test_evaluate_prophet_without_mlflow(mock_prophet_model, temp_test_csv):
    """MLflow 없이 Prophet 모델 평가 테스트"""
    os.environ['ENABLE_MLFLOW'] = 'false'

    with patch('src.utils.utils.model_dir') as mock_model_dir, \
            patch('joblib.load', return_value=mock_prophet_model):
        # 평가 시 사용하는 모델 경로를 mocking
        mock_model_dir.side_effect = lambda model_name=None: f'/tmp/{model_name}' if model_name else '/tmp'

        metrics = evaluate_prophet('/tmp/prophet_model.pkl', temp_test_csv, 'mock_run_id')

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0


def test_evaluate_sarimax_without_mlflow(temp_test_csv):
    """MLflow 없이 SARIMAX 모델 평가 테스트"""
    os.environ['ENABLE_MLFLOW'] = 'false'

    mock_sarimax_model = MagicMock()
    mock_sarimax_model.predict.return_value = np.array([15.0, 16.0, 17.0, 18.0, 19.0])

    with patch('src.utils.utils.model_dir') as mock_model_dir, \
            patch('joblib.load', return_value=mock_sarimax_model):
        mock_model_dir.side_effect = lambda model_name=None: f'/tmp/{model_name}' if model_name else '/tmp'

        metrics = evaluate_sarimax('/tmp/sarimax_model.pkl', temp_test_csv, 'mock_run_id')

        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0