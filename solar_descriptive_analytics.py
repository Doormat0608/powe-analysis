import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
# %matplotlib inline

data_folder_location = 'data/'

plant1_generation_data = pd.read_csv(
    data_folder_location + 'Plant_1_Generation_Data.csv',
    index_col=False, parse_dates=['DATE_TIME'], dayfirst=True)
plant2_generation_data = pd.read_csv(
    data_folder_location + 'Plant_2_Generation_Data.csv',
    index_col=False, parse_dates=['DATE_TIME'], dayfirst=True)

# Вывести таблицу данных о генерации электроэнергии на станции
print("Plant1 Generation Data Table --- -------", "\n Table Shape",
      plant1_generation_data.shape, "\n Table\n", plant1_generation_data.head(10))

# Импорт данных с погодных датчиков
plant1_weather_sensor_data = pd.read_csv(
    data_folder_location + 'Plant_1_Weather_Sensor_Data.csv', index_col=False)
plant2_weather_sensor_data = pd.read_csv(
    data_folder_location + 'Plant_2_Weather_Sensor_Data.csv', index_col=False)

# Вывести таблицу данных датчиков погоды на станции
print("Plant1 Weather Sensor Data Table --- -------", "\n Table Shape",
      plant1_weather_sensor_data.shape, "\n Table\n", plant1_weather_sensor_data.head(10))

# Статистика данных о генерации электроэнергии на станции 1
plant1_generation_data.describe()

# Статистика данных о генерации электроэнергии на станции 2
plant1_weather_sensor_data.describe()

# Нулевые значения во всем наборе данных
print("Общее количество нулевых значений в данных о генерации на станции 1: {}".format(
  plant1_generation_data.isnull().sum().sum()))
print("Общее количество нулевых значений в данных о генерации на станции 2: {}".format(
  plant2_generation_data.isnull().sum().sum()))
print("Общее количество нулевых значений в данных о погоде на станции 1: {}".format(
  plant1_weather_sensor_data.isnull().sum().sum()))
print("Общее количество нулевых значений в данных о погоде на станции 2: {}".format(
  plant1_weather_sensor_data.isnull().sum().sum()))

# Вывод количества инверторов в каждой электростанции
print("Общее количество инверторов на электростанции 1: {}".format(
  plant1_generation_data['SOURCE_KEY'].nunique()))
print("Общее количество инверторов на электростанции 2: {}".format(
  plant2_generation_data['SOURCE_KEY'].nunique()))

# Средняя постоянная (DC) и переменная (AC) мощность инверторов на электростанции 1 в порядке убывания
plant1_generation_data.groupby(['SOURCE_KEY'])['DC_POWER', 'AC_POWER'].mean(
).sort_values(by=['DC_POWER', 'AC_POWER'], ascending=False)

# Средняя постоянная (DC) и переменная (AC) мощность инверторов на электростанции 2 в порядке убывания
plant2_generation_data.groupby(['SOURCE_KEY'])['DC_POWER', 'AC_POWER'].mean(
).sort_values(by=['DC_POWER', 'AC_POWER'], ascending=False)

# Группировка данных о постоянной (DC) мощности генерации на электростанции 1 по часам и минутам
plant1_generation_data['DATE_TIME'] = pd.to_datetime(plant1_generation_data['DATE_TIME'], format='%d-%m-%Y %H:%M', errors='coerce')
plant1_generation_data['DATE_TIME'] = plant1_generation_data['DATE_TIME'].dt.floor('min')
times = pd.DatetimeIndex(plant1_generation_data['DATE_TIME'])
plant1_group_time = plant1_generation_data.groupby([times.hour, times.minute]).mean()['DC_POWER']

# Средняя выходная мощность постоянного тока (DC) для электростанции 1 в течение дня
plant1_group_time.plot(figsize=(20, 5))
plt.title('Средняя выходная мощность постоянного тока (DC) для электростанции 1 в течение дня')
plt.ylabel('Мощность DC (кВт)')

# Вычисление средней выходной мощности постоянного тока (DC) для каждого инвертора в разное время суток (час и минута)
plant1_group_inv_time = plant1_generation_data.groupby(
    [times.hour, times.minute, 'SOURCE_KEY']).DC_POWER.mean().unstack()
plant1_group_inv_time

# Построение и сравнение постоянного тока, подаваемого на инверторы первого завода
fig, ax = plt.subplots(ncols=3, nrows=1, dpi=200, figsize=(20, 5))
ax[0].set_title('Постоянный ток, подаваемый на первые 7 инверторов')
ax[1].set_title('Постоянный ток, подаваемый на следующие 8 инверторов')
ax[2].set_title('Постоянный ток, подаваемый на последние 7 инверторов')
ax[0].set_ylabel('Постоянный ток (кВт)')


plant1_group_inv_time.iloc[:, 0:7].plot(ax=ax[0], linewidth=5)
plant1_group_inv_time.iloc[:, 7:15].plot(ax=ax[1], linewidth=5)
plant1_group_inv_time.iloc[:, 15:22].plot(ax=ax[2], linewidth=5)

# Вычисление и построение графика эффективности инвертора для стацнии 1
plant1_group_inv = plant1_generation_data.groupby(['SOURCE_KEY']).mean()
plant1_group_inv['Inv_Efficiency'] = plant1_group_inv['AC_POWER'] * \
    100/plant1_group_inv['DC_POWER']

plant1_group_inv['Inv_Efficiency'].plot(figsize=(15, 5), style='o--')
plt.axhline(plant1_group_inv['Inv_Efficiency'].mean(),
            linestyle='--', color='green')
plt.title('График эффективности инверторов станции 1', size=20)
plt.ylabel('% эффективности')

# Вычисление и построение графика эффективности инвертора для станции 2
plant2_group_inv = plant2_generation_data.groupby(['SOURCE_KEY']).mean()
plant2_group_inv['Inv_Efficiency'] = plant2_group_inv['AC_POWER'] * \
    100/plant2_group_inv['DC_POWER']

plant2_group_inv['Inv_Efficiency'].plot(figsize=(15, 5), style='o--')
plt.axhline(plant2_group_inv['Inv_Efficiency'].mean(),
            linestyle='--', color='green')
plt.title('График эффективности инверторов стацнии 2', size=20)
plt.ylabel('% эффективности')

plant2_generation_Time = plant2_generation_data.groupby(
    ['DATE_TIME'], as_index=False).sum()
plant2_generation_Time

# Сохранение соответствующих данных
plant2_generation_Time = plant2_generation_Time[[
    'DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD']]
plant2_generation_Time

# Сохранение соответствующих данных
plant2_weather_sensor_data1 = plant2_weather_sensor_data.drop(
    ['PLANT_ID', 'SOURCE_KEY'], axis=1)

# Преобразование столбца 'DATE_TIME' в datetime64[ns]
plant2_weather_sensor_data1['DATE_TIME'] = pd.to_datetime(plant2_weather_sensor_data1['DATE_TIME'])

# Объединение данных о солнечной генерации станции и данных о погоде
merged_data = pd.merge(plant2_generation_Time, plant2_weather_sensor_data1, how='inner', on='DATE_TIME')

sns.pairplot(merged_data[['DC_POWER', 'AC_POWER', 'DAILY_YIELD',
             'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']])

merged_data_num = merged_data[['DC_POWER', 'AC_POWER', 'DAILY_YIELD',
                               'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
corr = merged_data_num.corr()

fig_dims = (2, 2)
sns.heatmap(round(corr, 2), annot=True, mask=(np.triu(corr, +1)))