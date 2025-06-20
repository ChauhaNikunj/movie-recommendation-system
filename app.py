import streamlit as st
import pickle
import pandas as pd

st.title('Movie Recommendation System')

# Load the movie dataframe and similarity matrix
movies_dict = pickle.load(open('movies.pkl', 'rb'))  # this should be a DataFrame or dict
movies_df = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function
def recommend(movie):
    try:
        index = movies_df[movies_df['title'] == movie].index[0]
    except IndexError:
        st.error(f"Movie '{movie}' not found in database.")
        return []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    for i in distances[1:6]:  # top 5 recommendations excluding the movie itself
        recommended_movies.append(movies_df.iloc[i[0]].title)

    return recommended_movies

# UI dropdown
selected_movie_name = st.selectbox(
    'Choose a Movie of your choice:',
    movies_df['title'].values
)

# On button click, show recommendations
if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)
