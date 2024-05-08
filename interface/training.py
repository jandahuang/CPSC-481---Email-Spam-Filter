#Import all libraries
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import string
# import seaborn as sns
# import plotly.graph_objects as go
from sklearn.decomposition import PCA
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
if not nltk.corpus.stopwords.words('english'):
    nltk.download('stopwords')
from nltk.corpus import stopwords
import json
warnings.filterwarnings('ignore')

# Preprocessing
#This function swaps 2 columns inside the dataframe
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

#This function removes punctuation from string
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

#Reading dataset
Email_dataset = pd.read_csv("spam_ham_dataset.csv")


s=Email_dataset["label"].value_counts()


#Dropping columns that are not needed
Email_dataset = Email_dataset.drop('Unnamed: 0', axis=1)
Email_dataset = Email_dataset.drop('label', axis=1)


#Creating a new feature, extracting subject of each email
subjects = []
for i in range(len(Email_dataset)):
    ln = Email_dataset["text"][i]
    line = ""
    for i in ln:
        if(i == '\r'):
            break
        line = line + i
    line = line.replace("Subject" , "")
    subjects.append(line)


Email_dataset['Subject'] = subjects


#Renaming the dataframe columns
Email_dataset.columns = ["Email_text" , "Labels" , "Email_Subject"]


#Swapping the dataframe columns 
Email_dataset = swap_columns(Email_dataset, 'Labels', 'Email_Subject')


#Converting all strings to lowercase
Email_dataset['Email_Subject'] = Email_dataset['Email_Subject'].str.lower()
Email_dataset['Email_text'] = Email_dataset['Email_text'].str.lower()


#Removing Punctuation from the data
Email_dataset['Email_Subject'] = Email_dataset['Email_Subject'].apply(remove_punctuations)
Email_dataset['Email_text'] = Email_dataset['Email_text'].apply(remove_punctuations)


#Creting seprate dataset for Spam and Non Spam emails, to perform analysis 
Spam = pd.DataFrame(columns = ['Email_text', 'Email_Subject', 'Labels'])
Non_Spam = pd.DataFrame(columns = ['Email_text', 'Email_Subject', 'Labels'])


#Creating Non_Spam email dataset 
for i in range(len(Email_dataset)):
    if(Email_dataset['Labels'][i] == 0):
        new_row = {'Email_text':Email_dataset['Email_text'][i], 'Email_Subject':Email_dataset['Email_Subject'][i], 'Labels':Email_dataset['Labels'][i]}
        Non_Spam.loc[len(Non_Spam)] = new_row


#Creating Spam email dataset 
for i in range(len(Email_dataset)):
    if(Email_dataset['Labels'][i] == 1):
        new_row = {'Email_text':Email_dataset['Email_text'][i], 'Email_Subject':Email_dataset['Email_Subject'][i], 'Labels':Email_dataset['Labels'][i]}
#         Spam = Spam.append(new_row, ignore_index=True)
        Spam.loc[len(Spam)] = new_row


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(h, y):
    return (-y * np.log(h + 1e-10) - (1 - y) * np.log(1 - h + 1e-10)).mean()


def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]


def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient


X = Email_dataset['Email_text']
y = Email_dataset['Labels'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit the CountVectorizer on your training data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()

# Transform your test data using the same vectorizer
X_test = vectorizer.transform(X_test).toarray()


num_iter = 1000
learning_rate = 0.1
weight = np.zeros(X_train.shape[1])


costs = []


# Logistic Regression



# Modified Naïve Bayes
total_emails = len(Email_dataset)

#prior probabilities
spam_not_spam_counter = Counter(Email_dataset["Labels"])
prior_not_spam = spam_not_spam_counter[0] / total_emails
prior_spam = spam_not_spam_counter[1] / total_emails
# print(prior_not_spam)
# print(prior_spam)

stop_words = set(stopwords.words('english'))
for word in stop_words:
    word = remove_punctuations(word)

custom_stop_words = {
    "gave",
    "see",
    "etc"
}

stop_words.update(custom_stop_words)

def remove_stopwords(body):
    return ' '.join([word for word in body.split() if word not in stop_words and not word.isdigit() and not re.match(r'(\w)\1{1,}', word) and not len(word) <= 2])

def remove_single_words(set, set2):
    text = ' '.join(set2)
    text = text.split()
    word_counts = Counter(text)
    single_occurrence_words = {text for text, count in word_counts.items() if count <= 35}
    return [' '.join(filter(lambda text: text not in single_occurrence_words, set.split())) for set in set]

def count_and_combine(set):
    return dict(Counter("".join(set).split()))


def smoothing(set1, set2, smoothing:int, reg:int):
    first_set = set("".join(set1).split())
    second_set = set("".join(set2).split())
    first_dict = count_and_combine(set1)
    second_dict = count_and_combine(set2)
    for word in first_dict:
        first_dict[word] += smoothing
    
    for word in second_dict:
        second_dict[word] += smoothing
    
    if first_set != second_set:
        for word in first_set:
            if word not in second_set:
                second_dict[word] = 1
        
        for word in second_set:
            if word not in first_set:
                first_dict[word] = 1
    
    temp_first_dict = first_dict.copy()
    for word in temp_first_dict:
        if first_dict[word] < reg and second_dict[word] < reg:
            del first_dict[word]
            del second_dict[word]

    temp_second_dict = second_dict.copy()
    for word in temp_second_dict:
        if second_dict[word] < reg and first_dict[word] < reg:
            del first_dict[word]
            del second_dict[word]
    return [first_dict, second_dict]

def calc_prob(set):
    set_len = len(set)
    set_total = 0

    for word in set:
        set_total += set[word]
    
    prob = set
    for word in set:
        prob[word] = set[word] / set_total
    
    return prob


# Remove stopwords from each paragraph in the list
first_non_spam_body = [remove_stopwords(body) for body in Non_Spam["Email_text"]]
first_spam_body = [remove_stopwords(body) for body in Spam["Email_text"]]
# print(first_non_spam_body)

smoothing_list = smoothing(first_non_spam_body, first_spam_body, 1, 20)
first_non_spam_body = smoothing_list[0]
first_spam_body = smoothing_list[1]

non_spam_prob = calc_prob(first_non_spam_body)
spam_prob = calc_prob(first_spam_body)
# for word in spam_prob:
#     print('"' + word + '": ' + str(spam_prob[word]) + ',')
naive_output = open('training/naivebayes.json', 'w')
naive_output.write(json.dumps({
    "non_spam": non_spam_prob,
    "spam": spam_prob,
    "prior_not_spam": prior_not_spam,
    "prior_spam": prior_spam,
}))
naive_output.close()

