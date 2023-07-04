import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
# %matplotlib inline

data_folder_location = 'data/'

# Импорт данных о генерации электроэнергии и данных датчиков погоды
plant1_generation_data = pd.read_csv(
    data_folder_location + 'Plant_1_Generation_Data.csv', index_col=False)
plant2_generation_data = pd.read_csv(
    data_folder_location + 'Plant_2_Generation_Data.csv', index_col=False)

# Импорт данных датчиков погоды
plant1_weather_sensor_data = pd.read_csv(
    data_folder_location + 'Plant_1_Weather_Sensor_Data.csv', index_col=False)
plant2_weather_sensor_data = pd.read_csv(
    data_folder_location + 'Plant_2_Weather_Sensor_Data.csv', index_col=False)

# Сохранение только необходимых данных
plant2_generation_Time = plant2_generation_data.groupby(
    ['DATE_TIME'], as_index=False).sum()
plant2_generation_Time = plant2_generation_Time[[
    'DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD']]

# Сохранение только необходимых данных
plant2_weather_sensor_data1 = plant2_weather_sensor_data.drop(
    ['PLANT_ID', 'SOURCE_KEY'], axis=1)

# Объединение данных о солнечной генерации установки 2 и данных о погоде
merged_data_plant2 = pd.merge(
    plant2_generation_Time, plant2_weather_sensor_data1, how='inner', on='DATE_TIME')

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import check_array
import xgboost as xgb
from xgboost import plot_importance
plt.style.use('fivethirtyeight')

target = merged_data_plant2['AC_POWER']
features = merged_data_plant2[['IRRADIATION', 'AMBIENT_TEMPERATURE']]

# Разделение данных на обучающую и тестовую выборки
#scaler = MinMaxScaler(feature_range=(0, 1))
#features_norm = scaler.fit_transform(features)
#target_norm = scaler.fit_transform(target)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=5)

#Прогнозирование с помощью линейной регрессии
lm = linear_model.LinearRegression()
model_lm = lm.fit(X_train, y_train)
pred_y_test_lm = lm.predict(X_test)

#  Вычисление MSE, MAE и MAPE для оценки ошибки модели
R2_lm = r2_score(y_test, pred_y_test_lm)
mse_lm = mean_squared_error(y_test, pred_y_test_lm, squared=False)
mae_lm = mean_absolute_error(y_test, pred_y_test_lm)


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[1:], y_train[:-1])
    return np.mean(np.abs(e_t / scale))


mase_lm = mean_absolute_scaled_error(y_test, pred_y_test_lm, y_train)


print('R2 с использованием линейной регрессии:', R2_lm, ' ' 'RMSE с использованием линейной регрессии:', mse_lm,
'\n ' 'MAE с использованием линейной регрессии:', mae_lm, ' ' 'MASE с использованием линейной регрессии:', mase_lm)

config = xgb.get_config()
config

# Обучение модели XG Boost на наборе данных о генерации солнечной энергии
model_xgb = xgb.XGBRegressor(n_estimators=50)
model_xgb.fit(X_train, y_train,
              eval_set=[(X_train, y_train),
                        (X_test, y_test)],
              early_stopping_rounds=50,
              verbose=False)

# Построение графика значимости признаков для прогнозирования
_ = plot_importance(model_xgb, height=0.5)

pred_y_test_xgb = model_xgb.predict(X_test)

# Вычисление MSE (среднеквадратичная ошибка), MAE (средняя абсолютная ошибка) и MAPE (средняя абсолютная процентная ошибка) для прогнозируемых значений с целью оценки ошибки модели
R2_xgb = r2_score(y_test, pred_y_test_xgb)
mse_xgb = mean_squared_error(y_test, pred_y_test_xgb, squared=False)
mae_xgb = mean_absolute_error(y_test, pred_y_test_xgb)
mase_xgb = mean_absolute_scaled_error(y_test, pred_y_test_xgb, y_train)

print('R2 с использованием XGBoost:', R2_xgb, ' ' 'RMSE с использованием XGBoost:', mse_xgb,
'\n ' 'MAE с использованием XGBoost:', mae_xgb, ' ' 'MASE с использованием XGBoost:', mase_xgb)

# Определение конвейера (Pipeline) и сетки параметров (Parameter grid)
pipeline = Pipeline([
    ('model', model_xgb)
])

param_grid = {
    'model__max_depth': [2, 3, 5, 7],
    'model__n_estimators': [10, 50, 100],
    'model__learning_rate': [0.02, 0.05, 0.1, 0.3],
    'model__min_child_weight': [0.5, 1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# Обучение модели
grid.fit(X_train, y_train)

# Вывод лучших параметров для модели, определенных с помощью Gridsearch
print(f"Лучшие параметры: {grid.best_params_}")

# Прогнозирование с помощью Gridsearch
pred_y_test_xgb_grid = grid.predict(X_test)

# Вычисление MSE (среднеквадратичная ошибка), MAE (средняя абсолютная ошибка) и MAPE (средняя абсолютная процентная ошибка) для прогнозируемых значений с целью оценки ошибки модели
R2_xgb_grid = r2_score(y_test, pred_y_test_xgb_grid)
mse_xgb_grid = mean_squared_error(y_test, pred_y_test_xgb_grid, squared=False)
mae_xgb_grid = mean_absolute_error(y_test, pred_y_test_xgb_grid)
mase_xgb_grid = mean_absolute_scaled_error(
    y_test, pred_y_test_xgb_grid, y_train)

print('R2 с использованием модели XGB_grid:', R2_xgb_grid, ' ' 'RMSE с использованием модели XGB_grid:', mse_xgb_grid,
'\n ' 'MAE с использованием модели XGB_grid:', mae_xgb_grid, ' ' 'MASE с использованием модели XGB_grid:', mase_xgb_grid)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# Разделение нормализованных данных на обучающую и тестовую выборки
scaler = MinMaxScaler(feature_range=(0, 1))
features_norm = scaler.fit_transform(features)
#target_norm = scaler.fit_transform(target)
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    features_norm, target, test_size=0.3, random_state=5)

y_test

pred_y_test_lm

pred_y_test_xgb

# Визуализация исходных данных и предсказанных значений
plt.plot(y_test, label='Исходные данные')
plt.plot(pred_y_test_lm, label='Предсказанные значения (Линейная регрессия)')
plt.plot(pred_y_test_xgb, label='Предсказанные значения (XG Boost)')
plt.legend()
plt.show()