#!/usr/bin/env python
# coding: utf-8

# # Uber Pick-up Demand Prediction using LSTM

# ### CPSC8650 Data Mining
# ### Spring 2021, Clemson University

# #### Nirali Bandaru, Rohan Gangisetty, Rajesh Kandimalla 

# In[1]:


#Importing the necessary libraries
import pandas as pd
import numpy as np
import time
import seaborn as sb


# In[2]:


#Importing data sets as dataframes using Pandas
#Importing data sets as dataframes using Pandas
apr_df = pd.read_csv(r"C:\Users\niral\OneDrive\Documents\Clemson\Semester 4\Data Mining\Project\Uber\Data\modified data2\uber-raw-data-apr14.csv")
may_df = pd.read_csv(r"C:\Users\niral\OneDrive\Documents\Clemson\Semester 4\Data Mining\Project\Uber\Data\modified data2\uber-raw-data-may14.csv")
jun_df = pd.read_csv(r"C:\Users\niral\OneDrive\Documents\Clemson\Semester 4\Data Mining\Project\Uber\Data\modified data2\uber-raw-data-jun14.csv")
jul_df = pd.read_csv(r"C:\Users\niral\OneDrive\Documents\Clemson\Semester 4\Data Mining\Project\Uber\Data\modified data2\uber-raw-data-jul14.csv")
aug_df = pd.read_csv(r"C:\Users\niral\OneDrive\Documents\Clemson\Semester 4\Data Mining\Project\Uber\Data\modified data2\uber-raw-data-aug14.csv")
sep_df = pd.read_csv(r"C:\Users\niral\OneDrive\Documents\Clemson\Semester 4\Data Mining\Project\Uber\Data\modified data2\uber-raw-data-sep14.csv")


# In[3]:


#EXPLORING and VIEWING THE IMPORTED DATAFRAMES

#APRIL
print("April: \n", apr_df.head())
print("Number of rows: ", apr_df.count())
print("Null values: ", apr_df.isnull().sum())

#MAY
print("May: \n", may_df.head())
print("Number of rows: ", may_df.count())
print("Null values: ", may_df.isnull().sum())

#JUN
print("Jun: \n", jun_df.head())
print("Number of rows: ", jun_df.count())
print("Null values: ", jun_df.isnull().sum())

#JUL
print("Jul: \n", jul_df.head())
print("Number of rows: ", jul_df.count())
print("Null values: ", jul_df.isnull().sum())

#AUG
print("Aug: \n", aug_df.head())
print("Number of rows: ", aug_df.count())
print("Null values: ", aug_df.isnull().sum())

#SEP
print("Sep: \n", sep_df.head())
print("Number of rows: ", sep_df.count())
print("Null values: ", sep_df.isnull().sum())

total_number_of_rows = 4534327


# In[4]:


#Creating a list of all dataframes to combine
df_list = [apr_df, may_df, jun_df, jul_df, aug_df, sep_df]

#Concatenate dataframes with pandas
dataframe = pd.concat(df_list)


# In[5]:


#Verify number of rows and format with count() and head(), respectively
dataframe.head()
dataframe.count()


# In[6]:


#Write new dataframe as combined csv file
dataframe.to_csv("combined_new_march7.csv")

"""
NOTE: There are a total of 4534327 rows in the combined dataframe, and excel can only display a maximum of
1,048,576 rows. To view all data, open file in Notepad++

"""


# ## Data Cleaning 

# In[7]:


#Checking for missing values

dataframe.isnull().sum()


# In[8]:


#Printing missing values

null_data = dataframe[dataframe.isnull().any(axis=1)]
print(null_data)


# In[9]:


dataframe = dataframe.dropna()
dataframe.head()


# In[10]:


#Verify that there are no missing values

count_of_null_values_total = dataframe.isnull().values.sum()
count_of_null_values_for_each_column = dataframe.isnull().sum()
boolean_result_for_null_values = dataframe.isnull().values.any()

print("Total null values: ", count_of_null_values_total)
print("Count of null values for each column: ", count_of_null_values_for_each_column)
print("Do null values exist? ", boolean_result_for_null_values)


# In[11]:


#Changing dtype object to appropriate data types for each column

dataframe['Date'] = pd.to_datetime(dataframe['Date'], errors = "coerce")
dataframe['Time'] = pd.to_datetime(dataframe['Time'], errors = "coerce").dt.time
#dataframe['Time'] = [time.time() for time in dataframe['Time']]
dataframe['Lat'] = pd.to_numeric(dataframe['Lat'], errors = "coerce")
dataframe['Lon'] = pd.to_numeric(dataframe['Lon'], errors = "coerce")
dataframe['Base'] = dataframe['Base'].astype('string')
dataframe.head()


# In[12]:


#dataframe["Date"]=pd.DatetimeIndex(dataframe["Date"]).date


# In[13]:


#Checking mininum and maximum values for each attribute
print("ATTRIBUTE RANGES\n")
print("DATE    Min: ", dataframe["Date"].min(), "Max: ", dataframe["Date"].max())
print("TIME    Min: ", dataframe["Time"].min(), "Max: ", dataframe["Time"].max())
print("LAT     Min: ", dataframe["Lat"].min(), "Max: ", dataframe["Lat"].max())
print("LON     Min: ", dataframe["Lon"].min(), "Max: ", dataframe["Lon"].max())


#print("BASE    Min: ", dataframe["Base"].min(), "Max: ", dataframe["Base"].max())


# In[14]:


number_of_rows_reduced = len(dataframe.index)
print("Number of rows after data cleaning: ", number_of_rows_reduced)


# In[15]:


dataframe.head()


# ## Data Visualization

# In[16]:


import matplotlib.pyplot as plt
import datetime


# In[17]:


#VISUALIZATIONS
'''
3. Frequency table of pickups by month
4. Total number of pickups by day of the week (line graph per month)
5. Heatmap of total pickups over 6 months
8. Number of rides by hour of day
10. Number of rides by months and days (6 plots)

11. Impact of holidays on number of rides (visualize for each month individually)
13. Number of rides over 24 hours, comparison with weekday vs weekend, and note peak hours for each.

.describe()
.summary()

'''


# In[18]:


plt.figure(figsize=(15, 15))
plt.plot(dataframe['Lon'], dataframe['Lat'], '.', ms=1, alpha=.5)
plt.xlim(-74.2, -73.7)
plt.ylim(40.7, 41)


# In[19]:


#Number of pickups by Month
dataframe['month'] = pd.DatetimeIndex(dataframe['Date']).month


# In[54]:


dataframe['month'].value_counts().plot(marker='o')
plt.xlabel('Month')
plt.ylabel('Number of Pickups')
plt.title('Pickups by Month')


# In[22]:


#Number of pickups by day of the week
dataframe['Date'] = pd.to_datetime(dataframe['Date'])


# In[23]:


dataframe['day_of_week'] = dataframe['Date'].dt.day_name()


# In[56]:


#Distribution of the pickups among the bases over those months
dataframe['Base'].value_counts().plot(marker='o')
plt.xlabel('Base')
plt.ylabel('Number of Pickups')
plt.title('Pickups by Base')


# In[28]:


dataframe.head()


# In[29]:


dataframe.hist("day_of_week","month",figsize=(30,15))


# In[28]:


# deep copy of the dataframe into df
df = dataframe.copy()
df.head()


# In[29]:


# Split the hour from time column
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

# Create a column for number of days using the Date column
df['Date'] = pd.to_datetime(df['Date'])
df['No_of_day'] = df['Date'].dt.day

df.head()


# In[49]:


# Average pickups by hours
average_no_day = df.groupby(['Date','Hour'])['Hour'].count()
average_no_day = average_no_day.groupby('Hour').agg([np.mean])

plt.figure(figsize = (12,5))
plt.plot(average_no_day, marker ='*',linewidth=2)
plt.xticks(np.arange(0,24))
plt.title('Average Pickups by Hour of Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Pickups')


# In[57]:


# Pie chart for the Uber Bases
fig = plt.figure(figsize=(6,6), dpi=200)
ax = plt.subplot(111, title='Pickups by Base')
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#99ff99']
dataframe.Base.groupby(dataframe.Base).count().plot(kind='pie', y='Base', colors = colors, autopct='%1.1f%%', shadow = False,startangle=90, fontsize=10)


# In[38]:


# Split the hour from time column
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

# Create a column for number of days using the Date column
df['Date'] = pd.to_datetime(df['Date'])
df['No_of_day'] = df['Date'].dt.day

# Create column that displays the day_name of the date column values
df['day_name'] = pd.DatetimeIndex(df['Date']).day_name()

# Create a column that shows the month of the date
df['Month'] = df['Date'].dt.month_name().str[:3]
df.head()

df.head()


# In[58]:


# Number of rides by hour of a day

plt.figure(figsize = (16,8))

sb.countplot(x='Hour', data = df, palette='viridis',saturation = 10)
sb.despine(bottom = True, left=True)

plt.xlabel('Hours 0-23')
plt.ylabel('Number of Pickups')
plt.title('Number of Pickups by Hours', fontsize = 14)


# In[61]:


# Average pickups by hours
average_no_day = df.groupby(['Date','Hour'])['Hour'].count()
average_no_day = average_no_day.groupby('Hour').agg([np.mean])

plt.figure(figsize = (12,5))
plt.plot(average_no_day, marker ='*',linewidth=2)
plt.xticks(np.arange(0,24))
plt.title('Average Pickups by Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Pickups')


# In[41]:


# Number of pickups by months

months = df.Month.value_counts().sort_values()
plt.figure(figsize = (10,8))

plt.plot(months.index, months.values, marker = '*', linewidth=2)
plt.xlabel('Months')
plt.ylabel('Pickups')
plt.title('Number of Pickups by Months',fontsize=14)


# In[62]:


# Number of rides by months and days

plt.figure(figsize = (14,12))

sb.countplot(x='Month',data=df,saturation=10,hue='day_name',hue_order =['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'])
sb.despine(bottom = True, left=True)
                                                                                     
plt.xlabel('Days of Each Month')
plt.ylabel('Number of Pickups')  
plt.title('Number of Pickups by Months and Days')
plt.legend(bbox_to_anchor = (1.03, 1))
                                           


# In[63]:


# Average pickups by hours
average_no_day = df.groupby(['Date','Hour'])['Hour'].count()
average_no_day = average_no_day.groupby('Hour').agg([np.mean])

plt.figure(figsize = (12,5))
plt.plot(average_no_day, marker ='*',linewidth=2)
plt.xticks(np.arange(0,24))
plt.title('Average Pickups by Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Pickups')


# In[64]:


#Freq of pickups at each hour grouped by month

df.groupby('Month')['day_of_week'].value_counts().unstack()


# In[65]:


df.groupby('Month')['day_of_week'].value_counts().plot(legend=True, kind='line')


# In[68]:


weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month = ['April', 'May', 'June', 'July', 'August', 'September']
Apr = [60861, 91185, 108631, 85067, 90303, 77218, 51251]
May = [63846, 76662, 89857, 128921, 133991, 102989, 56168]
Jun = [94655, 88134, 99654, 115325, 105056, 81364, 79656]
Jul = [93189, 137454, 147717, 148439, 102735, 90260, 76327]
Aug = [91633, 107124, 115255, 124117, 148673, 132225, 110246]
Sep = [137288, 163230, 135373, 153276, 160380, 162057, 116532]


# In[69]:


fig, ax = plt.subplots(figsize=(20,10))
ax.plot(weekday, Apr, marker="o")
ax.plot(weekday, May, marker = "o")
ax.plot(weekday, Jun, marker="o")
ax.plot(weekday, Jul, marker="o")
ax.plot(weekday, Aug, marker="o")
ax.plot(weekday, Sep, marker="o")
ax.legend(month)

plt.xlabel('Day of the Week')
plt.ylabel('Pickups')
plt.title('Number of Pickups Grouped by Month',fontsize=14)


# In[78]:


#dataframe['day_of_week'].value_counts().plot(marker='o')
#df.groupby('day_of_week', sort="False").size().plot(marker='o')
days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
i = df['Date'].dt.dayofweek.values.argsort()
df = df.iloc[i]
df['day_of_week'] = df['day_of_week'].map(days)
grouped_day_of_week = df.groupby('day_of_week', sort=False) 
grouped_day_of_week.plot(marker='o')

plt.xlabel('Day of the week')
plt.ylabel('Number of Pickups')
plt.title('Pickups by Day of the Week')


# In[85]:


df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
    ordered=True)
pickups_by_day = df['day_of_week'].value_counts()
print(pickups_by_day)
plt.xlabel('Day of the week')
plt.ylabel('Number of Pickups')
plt.title('Pickups by Day of the Week')
pickups_by_day.plot(marker='o', kind='line')

