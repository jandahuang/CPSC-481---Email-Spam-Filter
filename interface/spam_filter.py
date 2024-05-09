import pickle
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
import json
import re

# def vectorize(text_data):
#     preprocessed_data = text_data

#     # Create separate instances for train and test data
#     vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
#     vectorized_data = vectorizer.fit_transform(preprocessed_data)
#     return vectorized_data.toarray()

#     # preprocessed_data = [text.lower() for text in text_data]
#     # preprocessed_data = [remove_punctuations(text) for text in preprocessed_data]  # Apply remove_punctuations to each text entry
#     # vectorizer = CountVectorizer()
#     # vectorized_data = vectorizer.fit_transform(preprocessed_data)
#     # # vectorizer = TfidfVectorizer(max_features=45277)

#     # # vectorized_data = vectorizer.fit_transform(preprocessed_data)
#     # return vectorized_data.toarray()


# def predict():
#     with open('C:\\Users\\Janda Huang Sr\\OneDrive\\Desktop\\CPSC-481---Email-Spam-Filter\logistic_model.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#         input_text = ["hello this is a scam, give me all your money"]

#         vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
#         vectorized_data = vectorizer.fit_transform(input_text)
#         vectorized_data = vectorized_data.toarray()

#         prediction = loaded_model.predict(vectorized_data)  # Pass vectorized_data directly

#     return prediction
# predict()


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    with open(vectorizer_path, 'wb') as file:
        pickle.dump(vectorizer, file)

def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    with open(vectorizer_path, 'rb') as file:
        loaded_vectorizer = pickle.load(file)
    return loaded_model, loaded_vectorizer

def predict(input_text):
    loaded_model, loaded_vectorizer = load_model_and_vectorizer('C:\\Users\\Janda Huang Sr\\OneDrive\\Desktop\\CPSC-481---Email-Spam-Filter\logistic_model.pkl', 'vectorizer.pkl')
    
    #         loaded_model = pickle.load(file)
#         input_text = ["hello this is a scam, give me all your money"]

#         vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
#         vectorized_data = vectorizer.fit_transform(input_text)
#         vectorized_data = vectorized_data.toarray()

#         prediction = loaded_model.predict(vectorized_data)  # Pass vectorized_data directly

#     return prediction
    
    vectorized_data = loaded_vectorizer.transform(input_text)
    prediction = loaded_model.predict(vectorized_data)
    if (prediction[0]==1):
        print("Normal Mail")
    else:
        print("Spam Mail")


predict(["hello this is a scam, give me all your money"])
predict(["money money money money money money"])


# # Modified Naïve Bayes
# def naive_bayes(input_email):
#     # Load probs from json
#     naive_output = open('training/naivebayes.json')
#     spam_non_spam_prob = json.load(naive_output)
#     naive_output.close()
#     non_spam_prob = spam_non_spam_prob["non_spam"]
#     spam_prob = spam_non_spam_prob["spam"]
#     prior_not_spam = spam_non_spam_prob["prior_not_spam"]
#     prior_spam = spam_non_spam_prob["prior_spam"]
    
#     # Preprocess the input
#     input_email = input_email.lower()
#     input_email = re.sub(r'[^\w\s]', '', input_email).split()

#     total_spam_prob = prior_spam
#     for word in input_email:
#         if word in spam_prob:
#             total_spam_prob = total_spam_prob * spam_prob[word]

#     total_non_spam_prob = prior_not_spam
#     for word in input_email:
#         if word in non_spam_prob:
#             total_non_spam_prob = total_non_spam_prob * non_spam_prob[word]

#     return total_spam_prob >= total_non_spam_prob

# def spam_filter(input_email):
#     naivebayes = naive_bayes(input_email)
#     if naivebayes:
#         naivebayes = "spam"
#     else:
#         naivebayes = "not spam"

#     return {
#             "response": "not spam",
#             "tests": {
#                 "logistic_regression": "not spam",
#                 "naive_bayes": naivebayes,
#                 "decision_tree": "not spam",
#                 "support_vector_machine": "not spam",
#                 "k_nearest_neighbors": "not spam",
#                 "random_forest": "spam"
#             }
#         }