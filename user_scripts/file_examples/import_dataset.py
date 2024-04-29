import numpy as np
import pandas as pd


df = pd.read_json ("user_scripts/News_Category_Dataset_v3.json", lines = True)

categories = {'POLITICS': 0,
              'WELLNESS': 1,
              'ENTERTAINMENT':2,
              'TRAVEL': 3,
              'STYLE & BEAUTY': 4,
              'PARENTING': 5,
              'HEALTHY LIVING': 6,
              'QUEER VOICES': 7,
              'FOOD & DRINK': 8,
              'BUSINESS': 9}

### Extract only text and category columns ###
df = df[['headline', 'short_description', 'category']]

### Extract specific categories ###
df = df.loc[df['category'].isin(categories)]

### Extract up to k samples per category and replace category name with value ###
max_samples_per_category = 6500
for category in categories:
    category_df = df[df['category'] == category]
    category_df = category_df[:max_samples_per_category]
    other_categories_df = df[df['category'] != category]
    df = pd.concat([other_categories_df, category_df])
    df = df.replace({'category': {category: categories[category]}})

### shuffle dataset ###

df = df.sample(frac=1)

### convert to numpy arrays ###

news_headlines = np.array(df['headline'])
news_descriptions = np.array(df['short_description'])
news_topic_labels = np.array(df['category'])

print('news_headlines shape:', news_headlines.shape)          # (samples_per_cell,)
print('news_descriptions shape:', news_descriptions.shape)    # (samples_per_cell,)
print('news_topic_labels shape:', news_topic_labels.shape)    # (samples_per_cell,)


### clean texts ###

from project_utils.clean_text import clean_text_vec

print('Started text cleaning process...')
news_headlines = clean_text_vec(news_headlines)
news_descriptions = clean_text_vec(news_descriptions)
print('Text cleaning process has finished.')



### IMPORTANT PART ###

## input data must be a list of data, even if it is 1 numpy array
input_data = [news_headlines, news_descriptions]
input_labels = news_topic_labels







