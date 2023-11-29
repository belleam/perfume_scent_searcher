## Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


## Page configuration
st.set_page_config(page_title='Scent Searcher', page_icon='🌸', layout='wide')


## Load the data
@st.cache_data
def load_data(url):
    perfume_df = pd.read_csv('perfume_df.csv')
    return perfume_df
perfume_df = load_data('https://raw.githubusercontent.com/belleam/perfume_scent_searcher/main/perfume_df.csv')


## Construct TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
database = tfidf.fit_transform(perfume_df['Notes_clean'])


## Function for the recommendation
def scent_search(query):
    query_vec = tfidf.transform([query])
    scores = query_vec.dot(database.transpose())
    scores_array = scores.toarray()[0]
    sorted_indices = scores_array.argsort()[::-1]
    results = enumerate(sorted_indices[:5])
    perfume_indices = [i[1] for i in results]
    for i in perfume_indices:
        st.subheader(perfume_df['Name'].loc[i])
        st.write('Brand: ', perfume_df['Brand'].loc[i])
        st.write('Notes: ', perfume_df['Notes'].loc[i])
        st.write(' ')


## Page configuration
def main():

    ## Header
    scent_image = Image.open('scentsearch.png')
    col1, col2, col3 = st.columns([1, 1.5, 1])
    col2.image(scent_image, use_column_width=True)

    ## Get input from the user
    query = st.text_input('What are your favourite scent notes?', '')

    ## Launch recommendation function
    matchmake = ''
    if st.button('Meet your match'):
        matchmake = scent_search(query)

if __name__ == '__main__':
    main()