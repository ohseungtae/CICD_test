import fire
import pandas as pd
from icecream import ic
import os
from datetime import datetime

from src.data_loader.data_loader import collect_tokyo_weather
from src.preprocess.preprocess import load_and_split
from src.train.train import train_prophet, train_sarimax
from src.evaluate.evaluate import evaluate_prophet, evaluate_sarimax
from src.test.test import predict_future
from src.recommend.recommend import recommend_clothing
from src.utils.utils import init_seed
from src.config_loader import load_config
from src.model_select.modelselect import get_best_model
from src.serving.service import load_forecast, load_clothing

# MLflow 설정 - CI 환경에서는 비활성화
ENABLE_MLFLOW = os.getenv('ENABLE_MLFLOW', 'true').lower() == 'true'

if ENABLE_MLFLOW:
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000/")
    date_str = datetime.now().strftime("%Y-%m-%d")
    experiment_name = f"weather_prediction_{date_str}"
    mlflow.set_experiment(experiment_name)
else:
    experiment_name = f"weather_prediction_{datetime.now().strftime('%Y-%m-%d')}"


def run_all(config_path='config.yaml', model_name='prophet'):
    config = load_config(config_path)
    pipeline_cfg = config.get('pipeline', {})
    model_cfg = config.get(model_name, {})

    years = pipeline_cfg.get('years', 3)
    days = pipeline_cfg.get('days', 7)
    seed = pipeline_cfg.get('seed', 0)

    init_seed(seed)
    ic(f"Pipeline started with model: {model_name}")

    # 1. 데이터 수집
    csv_path = collect_tokyo_weather(years=years)

    # 2. 데이터 분할
    train_csv, test_csv = load_and_split()

    # 3. 모델 학습
    if model_name == 'prophet':
        model_path, run_id = train_prophet(train_csv, **model_cfg)
    elif model_name == 'sarimax':
        model_path, run_id = train_sarimax(train_csv, **model_cfg)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 4. 모델 평가
    if model_name == 'prophet':
        metrics = evaluate_prophet(model_path, test_csv, run_id)
    else:
        metrics = evaluate_sarimax(model_path, test_csv, run_id)

    print(f"[{model_name.upper()}] MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")

    # 4.1 모델 선택
    if ENABLE_MLFLOW:
        best_model = get_best_model(experiment_name)
    else:
        # CI 환경에서는 직접 모델 로드
        import joblib
        best_model = joblib.load(model_path)

    # 5. 미래 예측
    last_date = pd.read_csv(test_csv, parse_dates=['time'])['time'].max()
    forecast_filename = f"{model_name}_forecast.csv"
    future_csv = predict_future(best_model, last_date, days, save_name=forecast_filename)

    # fast api 서빙 - 기온
    load_forecast(future_csv)

    # 6. 옷차림 추천
    rec_path = recommend_clothing(future_csv=forecast_filename)
    ic(f"Pipeline complete, recommendations at {rec_path}")

    # fast api 서빙 - 옷
    load_clothing(rec_path)


# Fire CLI
if __name__ == '__main__':
    fire.Fire({
        'run_all': run_all,
        'collect_weather': collect_tokyo_weather,
        'split_data': load_and_split,
        'train_prophet': train_prophet,
        'train_sarimax': train_sarimax,
        'evaluate_prophet': evaluate_prophet,
        'evaluate_sarimax': evaluate_sarimax,
        'predict_future': predict_future,
        'recommend_clothing': recommend_clothing,
    })