#!/usr/bin/env python
# coding: utf-8

# In[8]:


# ----------------------------------------------------------------------------------------------
# The below two libraries are used to authenticate and authorize to connect to Google Sheets from Python
# -*- coding: utf-8 -*-
from google.auth import default
from googleapiclient.discovery import build
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# This library is used to read and write into Google Sheets from Python
import gspread
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# This library is used to maniputlate the extracted data from source.
import pandas as pd
import numpy as np
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# This is a regular expression library
import re
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


# In[9]:


# Authenticate using the application default credentials
try:
    credentials, project_id = default(scopes=['https://www.googleapis.com/auth/spreadsheets'])
except exceptions.DefaultCredentialsError:
    print("Error: Failed to retrieve application default credentials. Make sure you have authenticated with gcloud CLI using 'gcloud auth application-default login'.")
    exit()

# Build the service object using the credentials
service = build('sheets', 'v4', credentials=credentials)

# Access a specific spreadsheet
# Change it to correct worksheet Id.
spreadsheet_id = 'Work_Sheet_Id'


# Access a specific worksheet within the spreadsheet
sheet_name = 'Data'

# Read data from the worksheet using gspread
gspread_client = gspread.authorize(credentials)
spreadsheet = gspread_client.open_by_key(spreadsheet_id)
worksheet = spreadsheet.worksheet(sheet_name)
data = None
try:
    data = worksheet.get_all_values()
except Exception as e:
    print(f"Error getting values from worksheet: {e}")

# Convert the data to a Pandas DataFrame
if data is not None:
    # Create a DataFrame with the correct column names
    df = pd.DataFrame(data[1:], columns=data[0])
print(df)


# In[10]:


df.shape
# Change the path to original path
# df.to_csv('Path/desktop/reivew_data.csv', index = False)


# In[11]:


df.columns


# In[12]:


df.dtypes


# In[13]:


# change the path to original path.
df1=pd.read_csv('Path/reivew_data.csv', index_col = None, encoding='ISO-8859-1')
print(df1.info())
filtered_df = df1.copy()
df1.describe()

df1.head()


# In[43]:


df1.info()
# df = df1.copy()


# In[44]:


# Extracting dataframes each column values type.
def get_col_value_types(df):
    for col in df.columns:
        stacked = df[col].map(lambda x: type(x).__name__)
        type_counts = stacked.value_counts()
        type_percent = (type_counts / len(stacked) * 100).round(1)

        # Add a row for the total count and percentage
        type_counts_total = type_counts.copy()
        type_counts_total['Total'] = len(stacked)
        type_percent_total = (type_counts_total / len(stacked) * 100).round(1)

        # Add a row for the total count and percentage to the existing DataFrame
        results = pd.DataFrame({
            f'{col} Type': type_counts.index,
            f'{col} Count': type_counts.values,
            f'{col} Percentage': type_percent.values,
        })       
        print(results)
        print() 
        print('----------------------------------------------------------------------------------------')
        print(f"Total {col} Count: {sum(type_counts)}, and Total {col} %: {sum(type_percent)}")
        print('----------------------------------------------------------------------------------------')
        print()
        

get_col_value_types(df1)


# In[45]:


# Col Review - Group by Value vs Space only.
# Function to categorize text
dfreview = pd.DataFrame(df1['Review'])
categories = ['null', 'none', 'space', 'na or empty', 'filled_value'] 
def df_col_categorize(text):
    if pd.isna(text):
        return categories[0]
    elif isinstance(text, float) and math.isnan(text):
        return categories[0]
    elif text == 'None' or text.strip() == 'None':
        return categories[1]
    elif text.strip() == '':
        return categories[2]
    elif text.strip().lower() in ('na', 'nan'):
        return categories[3]
    else:
        return categories[4]
        
# Apply function to column
dfreview['Review_Category'] = df1['Review'].apply(df_col_categorize)
print(dfreview.groupby('Review_Category').size())
# print(dfreview)
# dfreview.apply(lambda x:  x.isspace() == True )
# result = dfreview.apply(lambda x: x.isspace() if isinstance(x, str) else False)
# print(result)


# In[46]:


import pandas as pd

dfdate = pd.DataFrame()

# Function to detect the date format
def detect_date_format(date_str):
    if pd.isnull(date_str) or not isinstance(date_str, str) or date_str.strip() == "":
        return None  # Covers NaN, None, NaT, Empty, Blank, '', ' '

    # Define regex patterns for date formats with day, month, and year
    date_patterns = {
        "%m-%d-%Y": r"^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-\d{4}$",
        "%m-%d-%Y": r"^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-\d{2}$",
        "%d-%m-%Y": r"^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-\d{4}$",
        "%d-%m-%Y": r"^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-\d{2}$"
    }

    # Check if the date string matches any of the patterns
    for fmt, pattern in date_patterns.items():
        if re.match(pattern, date_str):
            return fmt
    
    # If no match is found, return None
    return None

# dfdate = pd.DataFrame({
#     #'date': ['2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04'],
#     'date_format': ['%Y-%m-%d', None, '%Y-%m-%d', None]
# })

dfdate = pd.DataFrame(df1['date'])
dfdate['date_format'] = dfdate['date'].apply(detect_date_format)
print(dfdate.groupby('date_format').size())

# Create a new column to categorize the 'date_format' into 'Date' and 'Non-Date'
dfdate['date_category'] = dfdate['date_format'].apply(lambda x: 'Non-Date' if pd.isna(x) else 'Date')

# Group by the 'format_category' column and examine the groups
grouped = dfdate.groupby('date_category')
# Print each row info with Date or Non-Date.
# print(dfdate)
grouped.size()



# In[47]:


location_category = pd.DataFrame(df1['Location'])
location_category['Location_category'] = df1['Location'].apply(df_col_categorize)
location_category.groupby('Location_category').size()



# In[48]:


df_preprocess = filtered_df.copy()
df_preprocess

filtered_df


# In[49]:


df1


# In[50]:


(df1.shape)


# In[14]:


pd.options.display.max_rows = 6500


# In[15]:


filtered_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




