# import libraries
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# NLTK libraries
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

#Using Cosine Similarity
from sklearn.metrics.pairwise import pairwise_distances

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')

#Modelling 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb  # Load this xgboost

from collections import Counter
from imblearn.over_sampling import SMOTE

import joblib

data = pd.read_csv('sample30.csv' , encoding='latin-1')

def scrub_words(text):
    """Basic cleaning of texts."""
    
    # remove html markup
    text=re.sub("(<.*?>)"," ",text)
    
    # remove unneccessary words
    text = text.replace("!","")
    text = text.replace(":","")
    text = text.replace("_"," ")
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text = text.strip()
    text = re.sub(' +', ' ',text)
    
    return text

lemmatizer = nltk.stem.WordNetLemmatizer()
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        #print(word)
        #print(tag)
        #print("***************")
        if tag is None:
            #if there is no available tag, append the token as is
            #lemmatized_sentence.append(word)
            lemmatized_sentence.append(lemmatizer.lemmatize(word))
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        #print(lemmatized_sentence)
    return " ".join(lemmatized_sentence)

word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',    # Remove accents and perform other character normalization during the preprocessing step. 
    analyzer='word',            # Whether the feature should be made of word or character n-grams.
    token_pattern=r'\w{1,}',    # Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'
    ngram_range=(1, 3),         # The lower and upper boundary of the range of n-values for different n-grams to be extracted
    stop_words='english',
    sublinear_tf=True)

tmp_df = pd.DataFrame()
tmp_df['reviews_text']=data['reviews_text'].apply(lambda x: lemmatize_sentence(x))
tmp_df['name']=data['name']
tmp_df['reviews_username'] = data['reviews_username']

joblib.dump(tmp_df,  'lemmatize_sentence.pkl',compress=3)

lemma_temp = joblib.load('lemmatize_sentence.pkl')


data.dropna( how='any', subset=['user_sentiment'],inplace=True )
data['user_sentiment_updated'] = data['user_sentiment'].map({'Positive':1, 'Negative':0})

#converting into string
data['reviews_text'] = data['reviews_text'].astype('str')

# Remove punctuation 
data['reviews_text'] = data['reviews_text'].str.replace('[^\w\s]','')

# Remove Stopwords
stop = stopwords.words('english')
data['reviews_text'] = data['reviews_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Converting to lower case
data['reviews_text']= data['reviews_text'].str.lower()
data['reviews_text']=data['reviews_text'].apply(lambda x: scrub_words(x))
data['reviews_text']=data['reviews_text'].apply(lambda x: lemmatize_sentence(x))



#x=data['reviews_text'] 
#y=data['user_sentiment_updated']
seed = 50 
train, test = train_test_split(data, test_size=0.30, random_state=seed)
word_vectorizer.fit(train.reviews_text) 
## transforming the train and test datasets
X_train_transformed = word_vectorizer.transform(train.reviews_text.tolist())
X_test_transformed = word_vectorizer.transform(test.reviews_text.tolist())

counter = Counter(train.user_sentiment_updated)
sm = SMOTE()
# transform the dataset
X_train_transformed_sm, y_train_sm = sm.fit_resample(X_train_transformed, train.user_sentiment_updated)
counter = Counter(y_train_sm)

# Building the XGBoost Regularized Regression model

xgb_cfl = xgb.XGBClassifier(n_jobs = -1,objective = 'binary:logistic',n_estimators=1000,learning_rate = 0.1, min_child_weight = 1, gamma = 0.1,subsample = 1, colsample_bytree = 1, max_depth = 10)
xgb_cfl.fit(X_train_transformed_sm, y_train_sm) 

# Prediction Train Data
y_pred_train_sm= xgb_cfl.predict(X_train_transformed_sm)


# Prediction Test Data
y_pred_test = xgb_cfl.predict(X_test_transformed)

with open ('sentiment_model.pkl','wb') as fp:
    pickle.dump(xgb_cfl,fp)

df_pivot = train.pivot_table(index ='reviews_username', columns ='name', values ='reviews_rating', aggfunc='count').fillna(0)

# Copy the train dataset into dummy_train
dummy_train = train.copy()
dummy_train = dummy_train.groupby(['reviews_username','name'])['reviews_rating'].count().reset_index()
dummy_train['rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)
#dummy_train.head(2)
# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot(
    index='reviews_username',
    columns='name',
    values='rating'
).fillna(1)

user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
user_correlation[user_correlation<0]=0
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

joblib.dump(user_final_rating,  'recommend_model.pkl',compress=3)

recommend_model = joblib.load('recommend_model.pkl')

with open ('word_vectorizer.pkl','wb') as fp:
    pickle.dump(word_vectorizer,fp)

with open ('word_vectorizer.pkl','rb') as f:
    word_vectorizer = pickle.load(f)

with open ('sentiment_model.pkl','rb') as f:
    sentiment_model = pickle.load(f)

user_input = 'tony'
temp = pd.DataFrame(recommend_model.loc[user_input].sort_values(ascending=False))
if temp.shape[0] > 20:
    top20 =  temp[0:20]
else:
    top20 =  temp
    
top20 = top20.reset_index()
top20.rename(columns = {user_input:'score'}, inplace = True)
top20.insert(2, "Positive Sentiment(%)", "") 
top20.head()

for prod in top20['name']:
    #print(prod)
    rev = data[data['name'] == prod]
    if rev.shape[0] > 0:
        temp = rev['reviews_text'].apply(lambda x: lemmatize_sentence(x))
        temp1 = word_vectorizer.transform(temp)
        temp2 = sentiment_model.predict(temp1)
        pos = sum(temp2)
        total = len(temp2)
        percent = round(pos*100/total,2)
        top20.loc[top20['name'] == prod, ['Positive Sentiment(%)']] = percent

temp = top20.sort_values(by="Positive Sentiment(%)",ascending=False)#[0:5]
if temp.shape[0] > 5:
    top5 =  temp[0:5]
else:
    top5 =  temp
#top5.shape
top5['score'] = round(top5['score'],2)
top5.head()

