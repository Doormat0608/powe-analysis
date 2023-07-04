import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Import and Exploration

# In[3]:


data_folder_location = 'C:/Users/E0514808/Documents/Learning/Kaggle Competitions/Solar Power Generation/Data/'


# In[4]:


# Import Plant generation data and weather senson data
plant1_generation_data = pd.read_csv(
    data_folder_location +
    'Plant_1_Generation_Data.csv',
    index_col=False)
plant2_generation_data = pd.read_csv(
    data_folder_location +
    'Plant_2_Generation_Data.csv',
    index_col=False)


# In[5]:


# Print plant generation data table
print(
    "Plant1 Generation Data Table --- -------",
    "\n Table Shape",
    plant1_generation_data.shape,
    "\n Table\n",
    plant1_generation_data.head(10))
#print("\n Plant2 Generation Data Table --- -------","\n Table Shape", plant2_generation_data.shape,"\n Table\n", plant2_generation_data.head(10))


# In[6]:


# Import weather sensor data
plant1_weather_sensor_data = pd.read_csv(
    data_folder_location +
    'Plant_1_Weather_Sensor_Data.csv',
    index_col=False)
plant2_weather_sensor_data = pd.read_csv(
    data_folder_location +
    'Plant_2_Weather_Sensor_Data.csv',
    index_col=False)


# In[7]:


# Print plant weather sensor data table
print(
    "Plant1 Weather Sensor Data Table --- -------",
    "\n Table Shape",
    plant1_weather_sensor_data.shape,
    "\n Table\n",
    plant1_weather_sensor_data.head(10))
#print("\n Plant2 Weather Sensor Data Table --- -------","\n Table Shape", plant2_weather_sensor_data.shape,"\n Table\n", plant2_weather_sensor_data.head(10))


# In[8]:


# Statistics of Plant 1 Generation Data
plant1_generation_data.describe()


# In[9]:


# Statistics of Plant 2 Generation Data
plant1_weather_sensor_data.describe()


# In[10]:


# Null values in all the dataset
print("Total null values in plant 1 generation data is {}".format(
    plant1_generation_data.isnull().sum().sum()))
print("Total null values in plant 2 generation data is {}".format(
    plant2_generation_data.isnull().sum().sum()))
print("Total null values in plant 1 weather data is {}".format(
    plant1_weather_sensor_data.isnull().sum().sum()))
print("Total null values in plant 2 weather data is {}".format(
    plant1_weather_sensor_data.isnull().sum().sum()))


# In[11]:


# Print Number of Inverters in each Plant
print("Total inverters in plant 1 generation data are {}".format(
    plant1_generation_data['SOURCE_KEY'].nunique()))
print("Total inverters in plant 2 generation data are {}".format(
    plant2_generation_data['SOURCE_KEY'].nunique()))


# In[12]:


# Average DC and AC Power from Plant 1 Inverters in Descending order
plant1_generation_data.groupby(['SOURCE_KEY'])['DC_POWER', 'AC_POWER'].mean(
).sort_values(by=['DC_POWER', 'AC_POWER'], ascending=False)


# In[13]:


# Average DC and AC Power from Plant 2 Inverters in Descending order
plant2_generation_data.groupby(['SOURCE_KEY'])['DC_POWER', 'AC_POWER'].mean(
).sort_values(by=['DC_POWER', 'AC_POWER'], ascending=False)


# In[14]:


# Group plant 1 DC Power Generation by hour and minutes
times = pd.DatetimeIndex(plant1_generation_data.DATE_TIME)
plant1_group_time = plant1_generation_data.groupby(
    [times.hour, times.minute]).DC_POWER.mean()


# In[15]:
