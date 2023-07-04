# Прогнозирование выработки солнечной энергии с использованием линейной регрессии и XG Boost




from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
import keras
from xgboost import plot_importance
import xgboost as xgb
from sklearn.utils import check_array
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')





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

# Объединение данных о генерации солнечной энергии и данных о погоде (для второ
merged_data_plant2 = pd.merge(
    plant2_generation_Time,
    plant2_weather_sensor_data1,
    how='inner',
    on='DATE_TIME')


# Прогнозирование выходной мощности переменного тока (AC) электростанции




plt.style.use('fivethirtyeight')





target = merged_data_plant2['AC_POWER']
features = merged_data_plant2[['IRRADIATION', 'AMBIENT_TEMPERATURE']]





# Разделение данных на обучающую и тестовую выборки
#scaler = MinMaxScaler(feature_range=(0, 1))
#features_norm = scaler.fit_transform(features)
#target_norm = scaler.fit_transform(target)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=5)


# ## Prediction using Linear Regression




lm = linear_model.LinearRegression()
model_lm = lm.fit(X_train, y_train)
pred_y_test_lm = lm.predict(X_test)





# Обучение модели XG Boost на наборе данных о генерации солнечной энергии
R2_lm = r2_score(y_test, pred_y_test_lm)
mse_lm = mean_squared_error(y_test, pred_y_test_lm, squared=False)
mae_lm = mean_absolute_error(y_test, pred_y_test_lm)


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[1:], y_train[:-1])
    return np.mean(np.abs(e_t / scale))


mase_lm = mean_absolute_scaled_error(y_test, pred_y_test_lm, y_train)

print('R2 using Linear Regression:', R2_lm, '  '
      'RMSE using Linear Regression:', mse_lm, '\n '
      'MAE using Linear Regression:', mae_lm, '  '
      'MASE using Linear Regression:', mase_lm)


# Прогнозирование с использованием модели XG Boost

# In[13]:


config = xgb.get_config()
config
