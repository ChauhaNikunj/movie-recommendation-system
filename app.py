import streamlit as st
import pickle
import pandas as pd
import gdown
import os
import re

# === Page Title and Subtext ===
st.title("🎬 Movie Recommendation System")
st.caption("💡 Select a movie to get 5 similar recommendations.")
st.divider()

# === Load movies.pkl from local ===
movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies_df = pd.DataFrame(movies_dict)

# === Download and cache similarity.pkl ===
@st.cache_data
def load_similarity_with_gdown():
    file_path = 'similarity.pkl'
    if not os.path.exists(file_path):
        url = 'https://drive.google.com/uc?id=1vBkmwiYrzY_8tE8VQkQE8gellLHbtePY'
        gdown.download(url, file_path, quiet=False)

    with open(file_path, 'rb') as f:
        return pickle.load(f)

similarity = load_similarity_with_gdown()

# === Recommendation logic ===
def recommend(movie):
    try:
        index = movies_df[movies_df['title'] == movie].index[0]
    except IndexError:
        st.error(f"Movie '{movie}' not found in database.")
        return []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = [movies_df.iloc[i[0]].title for i in distances[1:6]]
    return recommended_movies

# === Sort dropdown titles alphabetically, ignoring symbols/case ===
def title_sort_key(title):
    return re.sub(r'^[^A-Za-z0-9]+', '', title).lower()

sorted_titles = sorted(movies_df['title'].values, key=title_sort_key)

# === UI Select Box ===
selected_movie_name = st.selectbox(
    '🎥 Choose a Movie:',
    sorted_titles,
    help="Start typing to filter the movie list."
)

# === Show Recommendations on Button Click ===
if st.button('🔍 Recommend'):
    if similarity is not None:
        recommendations = recommend(selected_movie_name)
        st.subheader("🎯 Top 5 Similar Movies")
        for i, rec in enumerate(recommendations, start=1):
            st.markdown(f"**{i}.** 🎞️ {rec}")
