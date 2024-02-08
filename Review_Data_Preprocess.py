#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This library is used to read and write into Google Sheets from Python
import gspread
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# This library is used to maniputlate the extracted data from source.
import pandas as pd
import numpy as np
# ----------------------------------------------------------------------------------------------
# This is a date parser library
from dateutil import parser
# ----------------------------------------------------------------------------------------------
# This library is used to extract location information from text.
from geotext import GeoText
# ----------------------------------------------------------------------------------------------
# This is a regular expression library
import re
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# This library is used to detect the text of the language
from langdetect import detect, DetectorFactory
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# This library is used to extract the language description from lanugage code. To fetch 
# description from two letter language code use pycountry and if it is a four letter code that 
# contains language code and region code then use babel's locale.
import pycountry
from babel import Locale
from textblob import TextBlob
# from cld2 import detect
# ----------------------------------------------------------------------------------------------
# The below library is used to extract country info from inconsistent location column
import geonamescache  # For gazetteer
import spacy  # For NER
# ----------------------------------------------------------------------------------------------
# This library is used to fuzzy match location
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
# ----------------------------------------------------------------------------------------------
# The below libraries are used to train and test the models
# # Import models
# ----------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification


import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
# import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize

# ----------------------------------------------------------------------------------------------
# !python -m spacy download en_core_web_sm - This is done - Downloaded and installed and need not to run this line. 
# ----------------------------------------------------------------------------------------------
import nltk
# import tensorflow as tf
# nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")


# In[122]:


get_ipython().run_line_magic('run', 'Review_Raw_Data_Analysis.py')


# In[181]:


df_preprocess = pd.read_csv('Path/reivew_data.csv',index_col=False, encoding='ISO-8859-1')
filtered_df = df_preprocess.copy()
# Column: review transformation

filtered_df = filtered_df.rename(columns = {
    'Review': 'review', 
    'Location': 'location'
    })

filtered_df.info()


# In[182]:


print(filtered_df['review'].isna().sum())
# filtered_df['review'] = filtered_df['review'].apply(lambda x: x.encode('utf-8').decode('utf-8') if isinstance(x, str) else x )
print(filtered_df.info())
def clean_text(text):
    # Remove non-printable characters and extra spaces  
    text = re.sub(r'[^[:print:]]+', '', text)    
    # Remove extra spaces and convert to lowercase     
    text = re.sub(r'\n+', '', text).lower() 
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]', '', text)
    return text

# Custom function to check if a string contains only spaces
def is_only_spaces(s):
    return s.isspace() and len(s.strip()) == 0
    

import pandas as pd

def clean_and_filter_text_column(df, text_column, clean_text_function):
    """
    Cleans and filters a text column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        text_column (str): The name of the text column to clean and filter.
        clean_text_function (function): A function that takes a string as input 
                                        and returns a cleaned string.

    Returns:
        None (modifies the DataFrame in-place)
    """

    # Apply Encoding and Decoding
    df[text_column].apply(lambda x: x.encode('utf-8').decode('utf-8') if isinstance(x, str) else x )
    
    # Apply cleaning function to text values
    df[text_column] = df[text_column].apply(lambda x: clean_text_function(x) if isinstance(x, str) else x)

    # Replace empty strings with NaN
    df[text_column] = df[text_column].str.strip().replace('', pd.NA)

    # Drop rows with NaN values
    df.dropna(subset=[text_column], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True) 
    
    

# filtered_df = filtered_df
clean_and_filter_text_column(filtered_df, 'review', clean_text)


print(filtered_df.info())
print(filtered_df['review'].isna().sum())





# In[184]:


pd.options.display.max_rows = 6500
# filtered_df.info()
filtered_df['date']


# In[185]:


# Colummn: Date Transfromation
# from dateutil import parser
# import pandas as pd

# Sample DataFrame with different date formats
# dfdt = pd.DataFrame({
#     'date': ['31-12-21', '01-01-2022', 'Mar-21', 'Apr-2021', 'Aug 2021']
# })

# Function to parse different date formats
def parse_date(date_str):
    # Handle complete dates (day-month-year)
    # , "%Y-%d-%m", "%y-%d-%m", "%y/%m/%d", "%Y/%m/%d"
    for fmt in ("%d-%m-%y", "%d-%m-%Y", "%m-%d-%y", "%m-%d-%Y", "%d/%m/%y", "%d/%m/%Y", "%m/%d/%y", "%m/%d/%Y", "%Y-%d-%m", "%y-%d-%m", 
                "%y/%m/%d", "%Y/%m/%d"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass

    # Handle month-year formats
    for fmt in ("%b-%y", "%b-%Y", "%b-%y%", "%b-%Y%", "%b %y", "%b %Y", "%b %y%", "%b %Y%" ):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    
    # Return None if all parsing attempts fail
    return None



filtered_df['date'] = filtered_df['date'].apply(parse_date)



# Print the DataFrame with 'date' column converted to datetime

print(filtered_df['date'])


# In[186]:


# Setting the seed for consistent language detection results
DetectorFactory.seed = 0

# Function to detect the language of a text
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'
    
def detect_language_text_blob(text): 
    try:
        return TextBlob(text)
    except:
        return 'unknown'


# review = filtered_df['review']
# review_df= pd.DataFrame(review)

# Add a new column with the detected language for each row
filtered_df['review_language'] = filtered_df['review'].apply(detect_language)

# filtered_df['review_language']


# In[187]:


filtered_df['iso_639-1_lang_code'] = pd.DataFrame(filtered_df[['review_language']])

def get_iso_639_1_code(lang_code):
#     clean_text(lang_code)
    if (lang_code.upper() == 'UNKNOWN'): return
#     if lang_code == None or lang_code.upper() == 'unknown': return 'unknown'
    try:
        # Attempt to parse the locale to handle complex codes (e.g., 'zh-cn')
        locale = Locale.parse(lang_code, sep='-')
        # Use the parsed locale to fetch the two-letter language code
        lang = pycountry.languages.get(alpha_2=locale.language)
        # Return the ISO 639-1 two-letter code
        return lang.alpha_2
#       return lang.alpha_2 if (len(lang.alpha_2) == 2 ) else 'unknown'
    except (ValueError, AttributeError):
        # Fallback for when the code cannot be parsed or matched
        # Attempt direct match with pycountry in case of simple codes
        try:
            lang = pycountry.languages.get(alpha_2=lang_code)
            return lang.alpha_2
#             return lang.alpha_2 if (len(lang.alpha_2) == 2 ) else 'unknown'
        except AttributeError:
            # Return original code if no match is found
            return lang_code
            

# Apply the function to map to ISO 639-1 codes
filtered_df['iso_639-1_lang_code'] = filtered_df['iso_639-1_lang_code'].apply(get_iso_639_1_code)

# print(filtered_df)


# In[188]:


# Define a function to extract language descriptions using pycountry
def get_language_description(code):
    if code == None: return
    language = pycountry.languages.get(alpha_2=code)
    return language.name if language else 'Unknown'

# Apply the function to the DataFrame column
filtered_df['language_description'] = filtered_df['iso_639-1_lang_code'].apply(get_language_description)





# In[189]:


filtered_df.head()


# In[190]:


# preprocessed_df = filtered_df[['review', 'date', 'iso_639-1_lang_code' , 'language_description', 'review_english']]
preprocessed_df = filtered_df.copy()
preprocessed_df.drop('location', axis = 1, inplace = True)
preprocessed_df.reset_index(drop = True, inplace = True)
preprocessed_df.info()


# In[191]:


preprocessed_df.head()


# In[192]:


preprocessed_df.info()


# In[193]:


len(preprocessed_df['review'].isna())


# In[194]:


print(filtered_df['language_description'].unique())
len(filtered_df['language_description'].value_counts())


# In[ ]:





# In[195]:


filtered_df['language_description'].value_counts()


# In[196]:


print(filtered_df['review'].isna().sum())
print(filtered_df['review'].isnull().sum())


# In[197]:


process_df = filtered_df.copy()


preprocessed_df.info()



# In[198]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string


# Drop unnecessary columns
preprocessed_df = preprocessed_df.drop(columns=['date', 'iso_639-1_lang_code', 'language_description'])

# 1. Remove stop words
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

preprocessed_df['review_english'] = preprocessed_df['review_english'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# 2. Remove punctuations
preprocessed_df['review_english'] = preprocessed_df['review_english'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# At this point, some 'review_english' could have turned into None (if they contained only stop words and punctuations).
# We replace these None with an empty string
preprocessed_df['review_english'] = preprocessed_df['review_english'].fillna('')

# 3. Tokenize
preprocessed_df['review_english'] = preprocessed_df['review_english'].apply(lambda x: word_tokenize(x))

# 4. Create DTM
count_vect = CountVectorizer()
dtm = count_vect.fit_transform(preprocessed_df['review_english'].apply(lambda x: ' '.join(x)))



# In[199]:


# dense_dtm = dtm.toarray()

# # convert to DataFrame
# dtm_df = pd.DataFrame(dense_dtm, columns=count_vect.get_feature_names_out())

# print(dtm_df)


# In[100]:


# Convert a subset of the sparse matrix to a DataFrame for inspection
subset_dtm_df = pd.DataFrame(dtm[:100].toarray(), columns=count_vect.get_feature_names_out())
print(subset_dtm_df)


# In[ ]:





# In[200]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

preprocessed_df['review_english'] = preprocessed_df['review_english'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])


# In[201]:


print(preprocessed_df['review_english'][3])


# In[202]:


# preprocessed_df[['review', 'review_english', 'iso_639-1_lang_code', 'language_description']]
original_df = process_df[['review', 'date', 'review_english', 'iso_639-1_lang_code' ,'language_description']].copy()

# original_df


# In[203]:


# Sentiment_Analysis with TextBlob and Vader


# Assuming preprocessed_df is your dataframe with the 'review_english' column preprocessed
# and 'date' column representing the date of each review

# Function for custom sentiment analysis using TextBlob
def get_sentiment_custom(text):
    if isinstance(text, list):
        text = ' '.join(text)  # Convert list of words to a sentence
    analysis = TextBlob(str(text))  # Ensure text is a string
    return analysis.sentiment.polarity


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Assuming preprocessed_df is your dataframe with the 'review_english' column preprocessed

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Function for custom sentiment analysis using VADER
def get_sentiment_vader(text):
    if isinstance(text, list):
        text = ' '.join(text)  # Convert list of words to a sentence
    analysis = analyzer.polarity_scores(str(text))  # Ensure text is a string
    return analysis['compound']

# Apply function to get sentiment score for each review
preprocessed_df['sentiment_score_vader'] = preprocessed_df['review_english'].apply(get_sentiment_vader)

preprocessed_df['sentiment_score_vader'] = preprocessed_df['sentiment_score_vader'].round(2)

# Now you can follow the same steps as in the original code to analyze the VADER sentiment scores.

# Apply function to get sentiment score for each review
preprocessed_df['sentiment_score_custom'] = preprocessed_df['review_english'].apply(get_sentiment_custom)

preprocessed_df['sentiment_score_custom'] = preprocessed_df['sentiment_score_custom'].round(2)
# print(combined_df.info())
# Assuming the indexes match, you can combine the dataframes
combined_df = original_df.join(preprocessed_df[['sentiment_score_custom', 'sentiment_score_vader']])
# print(combined_df.info())

# Convert 'date' column to DateTime format
combined_df['date'] = pd.to_datetime(combined_df['date'])

# Set the 'date' column as the index
combined_df.set_index('date', inplace=True)

# analyzing the sentiment scores
# print(combined_df.info())

# Insight 1: Average sentiment score over the entire dataset using custom sentiment analysis
average_sentiment_custom = combined_df['sentiment_score_custom'].mean()
average_sentiment_vader = combined_df['sentiment_score_vader'].mean()

# Insight 2: Average sentiment score by month using custom sentiment analysis
average_sentiment_by_month_custom = combined_df.resample('M')['sentiment_score_custom'].mean()
average_sentiment_by_month_vader = combined_df.resample('M')['sentiment_score_vader'].mean()
# Insight 3: Distribution of sentiment scores using custom sentiment analysis
sentiment_distribution_custom = combined_df['sentiment_score_custom'].value_counts(bins=5)
sentiment_distribution_vader = combined_df['sentiment_score_vader'].value_counts(bins=5)
# Insight 4: The day with the highest average sentiment score using custom sentiment analysis
best_day_custom = combined_df.groupby('date')['sentiment_score_custom'].mean().idxmax()
best_day_vader = combined_df.groupby('date')['sentiment_score_vader'].mean().idxmax()
# Insight 5: The day with the lowest average sentiment score using custom sentiment analysis
worst_day_custom = combined_df.groupby('date')['sentiment_score_custom'].mean().idxmin()
worst_day_vader = combined_df.groupby('date')['sentiment_score_vader'].mean().idxmin()

# Insight 6: Count of positive, neutral, and negative reviews
combined_df['sentiment_category'] = pd.cut(combined_df['sentiment_score_custom'],
                                           bins=[-1.01, -0.01, 0.01, 1],
                                           labels=['negative', 'neutral', 'positive'])
sentiment_counts_custom = combined_df['sentiment_category'].value_counts()


combined_df['sentiment_category_vader'] = pd.cut(combined_df['sentiment_score_vader'],
                                           bins=[-1.01, -0.01, 0.01, 1],
                                           labels=['negative', 'neutral', 'positive'])
sentiment_counts_vader = combined_df['sentiment_category_vader'].value_counts()

# Insight 7: Trends in sentiment over time (e.g., plotting sentiment over time)
# combined_df.groupby(combined_df['index'].dt.to_period('M'))['sentiment_score_custom'].mean().plot(kind='line')
combined_df.groupby(combined_df.index.to_period('M'), level=0)['sentiment_score_custom'].mean().plot(kind='line')

# Insight 8: Correlation between sentiment and other numerical features (if available)
# Replace the column names 'feature1', 'feature2' with your actual numerical features
# correlation = combined_df[['sentiment_score_custom', 'feature1', 'feature2']].corr()

# Insight 9: Sentiment score for specific keywords (e.g., filter reviews containing specific words)
keyword = 'good'  # Replace with your specific keyword
keyword_sentiment_custom = combined_df[combined_df['review'].str.contains(keyword)]['sentiment_score_custom']
keyword_sentiment_vader = combined_df[combined_df['review'].str.contains(keyword)]['sentiment_score_vader']
# Insight 10: Most positive and most negative reviews
most_positive_review_custom = combined_df.loc[combined_df['sentiment_score_custom'].idxmax()]['review_english']
most_negative_review_custom = combined_df.loc[combined_df['sentiment_score_custom'].idxmin()]['review_english']

most_negative_review_custom = combined_df.loc[combined_df['sentiment_score_custom'].idxmin()]['review_english']
most_positive_review_vader = combined_df.loc[combined_df['sentiment_score_vader'].idxmax()]['review_english']

# Calculate the count of positive, negative, and neutral sentiments
# combined_df['sentiment_category'] = pd.cut(combined_df['sentiment_score_custom'],
#                                            bins=[-1, -0.01, 0.01, 1],
#                                            labels=['negative', 'neutral', 'positive'])
# sentiment_counts = combined_df['sentiment_category'].value_counts()
# print(sentiment_counts)

# Display the counts of positive, negative, and neutral sentiments
print(sentiment_counts_custom)
print(sentiment_counts_vader)
print(average_sentiment_custom)
print(average_sentiment_vader)
# print(sentiment_counts_vader)
print(average_sentiment_by_month_custom)
print(average_sentiment_by_month_vader)
print(sentiment_distribution_custom)
print(sentiment_distribution_vader)
print(best_day_custom)
print(best_day_vader)
print(worst_day_custom)
print(worst_day_vader)
# print(sentiment_counts_custom)

# print(sentiment_counts_vader)
plt.savefig('MonthWiseSentimentScoreCustom')


# In[105]:


combined_df.head()
combined_df


# In[106]:


original_df.head()


# In[107]:


# pd.options.display.max_rows = 6500
preprocessed_df.head()


# In[108]:


print(average_sentiment_by_month_custom)


# In[109]:


null_score = preprocessed_df[preprocessed_df['sentiment_score_custom'].isnull()]
print(null_score)


# In[110]:


preprocessed_df.info()


# In[111]:


preprocessed_df[preprocessed_df['sentiment_score_custom'].round(2) == 0].head()


# In[112]:


# print(preprocessed_df.isna().any())

preprocessed_df['sentiment_score_custom'].round(2)


# In[113]:


print(preprocessed_df['sentiment_score_custom'].value_counts())


# In[114]:


preprocessed_df.isna().any()


# In[115]:


preprocessed_df.info()


# In[116]:


combined_df[combined_df['sentiment_score_custom'] == -1.0].head()


# In[117]:


combined_df.info()


# In[118]:


original_df.info()


# In[119]:


preprocessed_df.info()


# In[121]:


# test_result_df.info()


# In[204]:


test_result_df = combined_df.copy()

threshold = 0

# Binarize predictions  
test_result_df['label'] = test_result_df['sentiment_score_custom'].apply(lambda x: 1 if x>=threshold else 0)

# Group by review and aggregate predictions  
groups = test_result_df.groupby('review')['label'].agg(list)

# Count true/false positives and negatives
tp = len([1 for pred in groups if sum(pred)==len(pred)])  
tn = len([1 for pred in groups if sum(pred)==0])
fp = len([1 for pred in groups if (sum(pred)>0 and sum(pred)<len(pred))])
fn = len([1 for pred in groups if (sum(pred)==0 and sum(pred)<len(pred))])


conf_mat = [[tp, fp], 
            [fn, tn]]

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)  
recall = tp / (tp + fn)

print('Confusion Matrix:\n', conf_mat)
print('Accuracy: {}'.format(accuracy)) 
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))


# In[205]:


# Load the dataset
data = combined_df.copy()

# 1. Manual verification
sample_reviews = data.sample(n=20)
for index, row in sample_reviews.iterrows():
    print(f"Review: {row['review_english']}\nSentiment Score (Custom): {row['sentiment_score_custom']}\nSentiment Score (Vader): {row['sentiment_score_vader']}\n")

# 2. Cross-validation
X = data['review_english']
y = np.where(data['sentiment_score_custom'] >= 0, 1, 0)  # Convert sentiment scores to binary (positive or negative)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

y_pred = clf.predict(X_test_vectorized)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# 3. Visual inspection
sns.displot(data['sentiment_score_custom'], bins=20, kde=False)
plt.show()

wordcloud = WordCloud(background_color="white", width=800, height=800, max_words=150).generate(" ".join(data['review_english']))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[206]:


combined_df.info()


# In[207]:


plt.savefig("displot.png")


# In[208]:


combined_df.head()


# In[209]:


combined_df.to_csv('sentiment_Analysis_Result.csv')


# In[210]:


combined_df.head()


# In[211]:


combined_df.info()


# In[ ]:





# In[219]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample DataFrame with preprocessed and lemmatized text data
data = {'review_english': [["apple", "banana", "cherry"], ["banana", "orange", "grape"], ["apple", "cherry", "pear"]]}
df_similarity = pd.DataFrame(preprocessed_df['review_english'])

# Join the lemmatized words into a single string for each row
df_similarity['text_joined'] = df_similarity['review_english'].apply(lambda x: ' '.join(x))

# Create a CountVectorizer object to convert text data into a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_similarity['text_joined'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(X, X)
print("Cosine Similarity:")
print(cosine_sim)

# Calculate Jaccard similarity
def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1).union(list2))
    return intersection / union

jaccard_sim = [[jaccard_similarity(df_sim['review_english'].iloc[i], df_sim['review_english'].iloc[j]) for j in range(len(df_sim))] for i in range(len(df_sim))]
print("\nJaccard Similarity:")
print(jaccard_sim)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample DataFrame with preprocessed and lemmatized text data
# data = {'review_english': [["apple", "banana", "cherry"], ["banana", "orange", "grape"], ["apple", "cherry", "pear"]]}
df_sim = pd.DataFrame(prepprocessed_df)

# Join the lemmatized words into a single string for each row
df_sim['text_joined'] = df_sim['review_english'].apply(lambda x: ' '.join(x))

# Create a CountVectorizer object to convert text data into a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_sim['text_joined'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(X, X)
print("Cosine Similarity:")
print(cosine_sim)

# Calculate Jaccard similarity
def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1).union(list2))
    return intersection / union

jaccard_sim = [[jaccard_similarity(df_sim['review_english'].iloc[i], df_sim['review_english'].iloc[j]) for j in range(len(df_sim))] for i in range(len(df_sim))]
print("\nJaccard Similarity:")
print(jaccard_sim)


# In[218]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample DataFrame with preprocessed and lemmatized text data
# data = {'review_english': [["apple", "banana", "cherry"], ["banana", "orange", "grape"], ["apple", "cherry", "pear"]]}
df_similarity = pd.DataFrame(preprocessed_df['review_english'])

# Join the lemmatized words into a single string for each row
df_similarity['text_joined'] = df_similarity['review_english'].apply(lambda x: ' '.join(x))

# Create a CountVectorizer object to convert text data into a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_similarity['text_joined'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(X, X)
print("Cosine Similarity:")
print(cosine_sim)

# Calculate Jaccard similarity
def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1).union(list2))
    return intersection / union

jaccard_sim = [[jaccard_similarity(df_similarity['review_english'].iloc[i], df_similarity['review_english'].iloc[j]) for j in range(len(df_similarity))] for i in range(len(df_similarity))]
print("\nJaccard Similarity:")
print(jaccard_sim)


# In[ ]:


preprocessed_df['review_english'].head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




