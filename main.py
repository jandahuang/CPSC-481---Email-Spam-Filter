import pickle
import streamlit as st

model = pickle.load(open("classification", "rb"))
cv = pickle.load()


def main():
    st.title("Email Spam Filter")
    # activites = ["Classf"]

    
