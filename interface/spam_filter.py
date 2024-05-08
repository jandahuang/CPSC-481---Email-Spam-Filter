import pickle
import numpy as np
import string
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json
import re

# def vectorize(text_data):

    
#     preprocessed_data = [text.lower() for text in text_data]
#     preprocessed_data = [remove_punctuations(text) for text in preprocessed_data]  # Apply remove_punctuations to each text entry
#     vectorizer = CountVectorizer()
#     vectorized_data = vectorizer.fit_transform(preprocessed_data)
#     # vectorizer = TfidfVectorizer(max_features=45277)

#     # vectorized_data = vectorizer.fit_transform(preprocessed_data)
#     return vectorized_data.toarray()

# def remove_punctuations(text):
#     return text.translate(str.maketrans("", "", string.punctuation))

# def predict():
#     with open('logistic_model.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#         input_text = "hello this is a scam, give me all your money"
#         vectorized_input = vectorize([input_text])  # Pass input_text as a list
#         print("the vectorized input is:", vectorized_input)

#         predicted_class = loaded_model.predict(vectorized_input)

#         print(f"Predicted class: {predicted_class}")

# predict()




# Modified Naïve Bayes
def naive_bayes(input_email):
    # Load probs from json
    naive_output = open('training/naivebayes.json')
    spam_non_spam_prob = json.load(naive_output)
    naive_output.close()
    non_spam_prob = spam_non_spam_prob["non_spam"]
    spam_prob = spam_non_spam_prob["spam"]
    prior_not_spam = spam_non_spam_prob["prior_not_spam"]
    prior_spam = spam_non_spam_prob["prior_spam"]
    
    # Preprocess the input
    input_email = input_email.lower()
    input_email = re.sub(r'[^\w\s]', '', input_email).split()

    total_spam_prob = prior_spam
    for word in input_email:
        if word in spam_prob:
            total_spam_prob = total_spam_prob * spam_prob[word]

    total_non_spam_prob = prior_not_spam
    for word in input_email:
        if word in non_spam_prob:
            total_non_spam_prob = total_non_spam_prob * non_spam_prob[word]

    return total_spam_prob >= total_non_spam_prob

def spam_filter(input_email):
    naivebayes = naive_bayes(input_email)
    if naivebayes:
        naivebayes = "spam"
    else:
        naivebayes = "not spam"

    return {
            "response": "not spam",
            "tests": {
                "logistic_regression": "not spam",
                "naive_bayes": naivebayes,
                "decision_tree": "not spam",
                "support_vector_machine": "not spam",
                "k_nearest_neighbors": "not spam",
                "random_forest": "spam"
            }
        }