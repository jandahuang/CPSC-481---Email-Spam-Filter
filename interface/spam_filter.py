import pickle
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
import json
import re
import pandas as pd


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

def predict(input_text, model_name):
    model = "{}.pkl".format(model_name)
    loaded_model, loaded_vectorizer = load_model_and_vectorizer(model, 'vectorizer.pkl')
    
    inpput= list(input_text)
    vectorized_data = loaded_vectorizer.transform(inpput)
    prediction = loaded_model.predict(vectorized_data)
    return prediction
    # if (prediction[0]==1):
    #     print("Normal Mail")
    # else:
    #     print("Spam Mail")

# ====================
# For testing purposes
# ====================
# Reading dataset
# Email_dataset = pd.read_csv("spam_ham_dataset.csv")
# predict("among us sus among us sus among us sus, I am not sus", "logistic_model")

# message = string([Email_dataset['text'][4]])
# print(Email_dataset['text'][4])

# predict([Email_dataset['text'][4]], "logistic_model")

# predict(message, "logistic_model")
# ====================
# End of testing block
# ====================



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
    spam_count = 0
    ham_count = 0

    naivebayes_custom = naive_bayes(input_email)
    if naivebayes_custom:
        naivebayes_custom = "spam"
        spam_count += 1
    
    else:
        naivebayes_custom = "not spam"
        ham_count += 1
    
    
    
    logistic = predict(input_email, "logistic_model")
    if logistic[0]==1:
        logistic = "not spam"
        ham_count += 1
    else:
        logistic = "spam"
        spam_count += 1

    bayes = predict(input_email, "bayes_model")
    if bayes[0]==1:
        bayes = "not spam"
        ham_count += 1
    else:
        bayes = "spam"
        spam_count += 1
        

    decision_tree = predict(input_email, "decision_tree_model")
    if decision_tree[0]==1:
        decision_tree = "not spam"
        ham_count += 1
    else:
        decision_tree = "spam"
        spam_count += 1

    random_forest = predict(input_email, "random_forest_model")
    if random_forest[0]==1:
        random_forest = "not spam"
        ham_count += 1
    else:
        random_forest = "spam"
        spam_count += 1

    support_vector_model = predict(input_email, "support_vector_model")
    if support_vector_model[0]==1:
        support_vector_model = "not spam"
        ham_count += 1
    else:
        support_vector_model = "spam"
        spam_count += 1

    knn = predict(input_email, "knn_model")
    if knn[0]==1:
        knn = "not spam"
        ham_count += 1
    else:
        knn = "spam"
        spam_count += 1

    if spam_count > ham_count:
        result = "spam"
    else:
        result = "not spam"

    return {
            "response": result,
            "tests": {
                "logistic_regression": logistic,
                "naive_bayes_custom": naivebayes_custom,
                "naive_bayes" : bayes, 
                "decision_tree": decision_tree,
                "support_vector_machine": support_vector_model,
                "k_nearest_neighbors": knn,
                "random_forest": random_forest
            }
        }

# print(spam_filter(["among us sus among us sus among us sus, I am not sus"]))

Email_dataset = pd.read_csv("spam_ham_dataset.csv")
message = [Email_dataset['text'][4]]
print(spam_filter(message))
