#Import all libraries
import numpy as np
import pandas as pd
import nltk
import pickle
# import matplotlib.pyplot as plt
import string
# from sklearn.decomposition import PCA
import warnings
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
# from sklearn.preprocessing import LabelEncoder
from collections import Counter
if not nltk.corpus.stopwords.words('english'):
    nltk.download('stopwords')
from nltk.corpus import stopwords
import json
import re
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
df = pd.read_csv("spam_ham_dataset.csv")



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


#Converting all strings to lowercase
Email_dataset['Email_Subject'] = Email_dataset['Email_Subject'].str.lower()
Email_dataset['Email_text'] = Email_dataset['Email_text'].str.lower()


#Removing Punctuation from the data
Email_dataset['Email_Subject'] = Email_dataset['Email_Subject'].apply(remove_punctuations)
Email_dataset['Email_text'] = Email_dataset['Email_text'].apply(remove_punctuations)


X = df['text']
y = df['label_num'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer

# Create separate instances for train and test data
vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)


X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test) # Use transform, not fit_transform

y_train = y_train.astype('int')
y_test = y_test.astype('int')



with open('./training/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


from sklearn.linear_model import LogisticRegression
Logistic_model = LogisticRegression(random_state = 0)
Logistic_model.fit(X_train_features, y_train)


y_pred = Logistic_model.predict(X_train_features)



y_pred = Logistic_model.predict(X_test_features)





with open('./training/logistic_model.pkl', 'wb') as file:
    pickle.dump(Logistic_model, file)
    

message = df['text'][8] 
df




from sklearn import feature_extraction

# Input = Email_dataset['Email_text'][4]  
Input = Email_dataset['Email_text'][10] 
my_list = Input.split(",")


input_text_features = vectorizer.transform(my_list)
prediction = Logistic_model.predict(input_text_features)







from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(random_state = 0)
decision_tree_model.fit(X_train_features, y_train)

y_pred = decision_tree_model.predict(X_train_features)






y_pred = Logistic_model.predict(X_test_features)






with open('./training/decision_tree_model.pkl', 'wb') as file:
    pickle.dump(decision_tree_model, file)



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
naive_output = open('./training/naivebayes.json', 'w')
naive_output.write(json.dumps({
    "non_spam": non_spam_prob,
    "spam": spam_prob,
    "prior_not_spam": prior_not_spam,
    "prior_spam": prior_spam,
}))
naive_output.close()




# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

bayes_model = MultinomialNB()
bayes_model.fit(X_train_features, y_train)


y_pred = bayes_model.predict(X_train_features)






y_pred = bayes_model.predict(X_test_features)







with open('./training/bayes_model.pkl', 'wb') as file:
    pickle.dump(bayes_model, file)


from sklearn.svm import SVC
support_vector_model = SVC(kernel='rbf', random_state=42)
support_vector_model.fit(X_train_features, y_train)


y_pred = support_vector_model.predict(X_train_features)






y_pred = support_vector_model.predict(X_test_features)









with open('./training/support_vector_model.pkl', 'wb') as file:
    pickle.dump(support_vector_model, file)



# KNN
from sklearn.neighbors import KNeighborsClassifier
k = 5
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_features, y_train)


y_pred = knn_model.predict(X_train_features)






y_pred = knn_model.predict(X_test_features)







with open('./training/knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)



from sklearn.ensemble import RandomForestClassifier as RFR

random_forest_model = RFR(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_features, y_train)


y_pred = random_forest_model.predict(X_train_features)






y_pred = random_forest_model.predict(X_test_features)






with open('./training/random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest_model, file)



