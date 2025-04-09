# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
from surprise import Dataset, Reader, SVD
import heapq
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import kagglehub
from kagglehub import KaggleDatasetAdapter

# --- Configuration & Constants ---
DATA_FOLDER = 'Data'
MOVIES_RATINGS_FILE = os.path.join(DATA_FOLDER, 'movies_ratings.csv') 
NEW_MOVIES_FILE = os.path.join(DATA_FOLDER, 'updated_new_movies.csv') 
N_RECOMMENDATIONS = 10
N_NEW_MOVIES = 2
N_DISPLAY_USER_RATINGS = 5
N_SIMILAR_MOVIES_DISPLAY = 5
N_PCA_COMPONENTS = 2

# --- Data Downloading and Preprocessing ---
@st.cache_data
def download_and_preprocess_data():
    """Download datasets from Kaggle and preprocess them."""
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Download MovieLens datasets
    st.write("Downloading MovieLens datasets...")
    movies_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "shubhammehta21/movie-lens-small-latest-dataset",
        "movies.csv",
    )
    ratings_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "shubhammehta21/movie-lens-small-latest-dataset",
        "ratings.csv",
    )
    movies_df.to_csv(os.path.join(DATA_FOLDER, 'movies.csv'), index=False)
    ratings_df.to_csv(os.path.join(DATA_FOLDER, 'ratings.csv'), index=False)

    # Merge movies and ratings
    movie_ratings_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')
    movie_ratings_df.drop(columns=['timestamp'], inplace=True)
    movie_ratings_df.to_csv(MOVIES_RATINGS_FILE, index=False)
    st.write("MovieLens data merged and saved.")

    # Download and preprocess new movies dataset
    st.write("Downloading Top 100 Trending Movies 2025 dataset...")
    new_movies_df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "taimoor888/top-100-trending-movies-of-2025",
        "Top_100_Trending_Movies_2025.csv",
    )

    # Preprocessing new movies
    def parse_box_office(value):
        if pd.isna(value):
            return 0
        value = value.replace('$', '').strip()
        if 'billion' in value.lower():
            return float(value.replace(' billion', '')) * 1000
        elif 'million' in value.lower():
            return float(value.replace(' million', ''))
        return 0

    def assign_rating(box_office):
        if box_office >= 1000:
            return 4.5
        elif box_office >= 500:
            return 3.5
        else:
            return 2.5

    increament_value = 193609
    new_movies_df = new_movies_df[new_movies_df['Rank'].notna()]
    new_movies_df['Rank'] = (new_movies_df['Rank'].astype(int) + increament_value)
    new_movies_df['Genre'] = new_movies_df['Genre'].str.replace(',', '|')
    new_movies_df.rename(columns={'Rank': 'movieId', 'Genre': 'genres', 'Title': 'title', 'IMDB Rating': 'rating'}, inplace=True)
    new_movies_df['box_office_millions'] = new_movies_df['Box Office Prediction'].apply(parse_box_office)
    new_movies_df['rating'] = new_movies_df['rating'].fillna(new_movies_df['box_office_millions'].apply(assign_rating))
    new_movies_df = new_movies_df[['movieId', 'title', 'genres', 'rating']]
    new_movies_df.to_csv(NEW_MOVIES_FILE, index=False)
    st.write("New movies data preprocessed and saved.")

    return movie_ratings_df, new_movies_df

@st.cache_data
def load_data():
    """Loads movies_ratings and updated_new_movies data, preprocesses for compatibility."""
    try:
        movie_ratings_df, new_movies_df = download_and_preprocess_data()

        ratings_df = movie_ratings_df[['userId', 'movieId', 'rating']].copy()
        movies_df = movie_ratings_df[['movieId', 'title', 'genres']].drop_duplicates().copy()

        new_movies_df = new_movies_df[['movieId', 'title', 'genres']]
        movies_df = pd.concat([movies_df, new_movies_df], ignore_index=True).drop_duplicates(subset='movieId')

        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        movie_titles = movies_df.set_index('movieId')['title'].to_dict()
        movie_genres = movies_df.set_index('movieId')['genres_list'].to_dict()
        movie_ratings_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')
        new_movie_ids = new_movies_df['movieId'].tolist()

        st.success("Data loaded and preprocessed successfully!")
        return movies_df, ratings_df, movie_ratings_df, movie_titles, movie_genres, new_movie_ids
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None, None, None, None

@st.cache_resource
def train_svd_model_and_get_factors(_ratings_df, _movies_df):
    # [Your existing train_svd_model_and_get_factors function remains unchanged]
    if _ratings_df is None or _movies_df is None:
        st.warning("Cannot train model, data not loaded.")
        return None, None, None, None, None

    progress_text = "Training Recommendation Model (SVD) & Computing PCA... Please wait."
    my_bar = st.progress(0, text=progress_text)

    try:
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(_ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()

        my_bar.progress(25, text="Built trainset. Starting SVD training...")
        model = SVD(n_factors=100, n_epochs=20, biased=True, random_state=42, verbose=False)
        model.fit(trainset)

        my_bar.progress(60, text="SVD trained. Extracting factors...")
        item_factors = model.qi
        raw_to_inner_id_map = {trainset.to_raw_iid(inner_id): inner_id for inner_id in trainset.all_items()}
        inner_to_raw_id_map = {v: k for k, v in raw_to_inner_id_map.items()}

        num_items_in_trainset = trainset.n_items
        factors_for_pca = item_factors[:num_items_in_trainset]

        my_bar.progress(75, text="Factors extracted. Computing PCA...")
        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
        item_factors_pca = pca.fit_transform(factors_for_pca)

        pca_df = pd.DataFrame({
            'movieId_inner': list(inner_to_raw_id_map.keys()),
            'pca_comp_1': item_factors_pca[:, 0],
            'pca_comp_2': item_factors_pca[:, 1]
        })
        pca_df['movieId'] = pca_df['movieId_inner'].map(inner_to_raw_id_map)
        pca_df = pd.merge(pca_df, _movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')

        my_bar.progress(100, text="Model training & PCA complete!")
        st.success("Model trained and PCA computed!")
        my_bar.empty()

        return model, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, pca_df
    except Exception as e:
        my_bar.empty()
        st.error(f"An error occurred during model training or PCA: {e}")
        return None, None, None, None, None

# --- Helper Functions ---
def cosine_similarity(vec1, vec2):
    """Compute centered cosine similarity between two vectors to allow negative values."""
    # Center the vectors by subtracting their means
    vec1_centered = vec1 - np.mean(vec1)
    vec2_centered = vec2 - np.mean(vec2)
    norm_vec1 = np.linalg.norm(vec1_centered)
    norm_vec2 = np.linalg.norm(vec2_centered)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return np.dot(vec1_centered, vec2_centered) / (norm_vec1 * norm_vec2)

def get_similar_movies(target_movie_id, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, movie_genres, is_new_movie=False, n=5):
    """Get similar movies based on SVD factors (existing) or genre overlap (new)."""
    if is_new_movie:
        target_genres = set(movie_genres.get(target_movie_id, []))
        similarities = []
        for movie_id, genres in movie_genres.items():
            if movie_id == target_movie_id:
                continue
            sim = len(target_genres.intersection(set(genres))) / max(len(target_genres.union(set(genres))), 1)
            similarities.append((movie_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    else:
        if target_movie_id not in raw_to_inner_id_map:
            return []
        target_inner_id = raw_to_inner_id_map[target_movie_id]
        if target_inner_id >= len(item_factors):
            return []
        target_factor = item_factors[target_inner_id]
        similarities = []
        for inner_id in inner_to_raw_id_map.keys():
            if inner_id == target_inner_id or inner_id >= len(item_factors):
                continue
            factor = item_factors[inner_id]
            sim = cosine_similarity(target_factor, factor)
            raw_id = inner_to_raw_id_map[inner_id]
            similarities.append((raw_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

def get_user_genre_averages(user_id, ratings_df, movie_genres):
    """Calculate average ratings per genre for a user."""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    genre_sums = {}
    genre_counts = {}
    for _, row in user_ratings.iterrows():
        movie_id = row['movieId']
        rating = row['rating']
        genres = movie_genres.get(movie_id, [])
        for genre in genres:
            if genre != "(no genres listed)":
                genre_sums[genre] = genre_sums.get(genre, 0) + rating
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    genre_averages = {genre: genre_sums[genre] / genre_counts[genre] for genre in genre_sums if genre_counts[genre] > 0}
    overall_avg = user_ratings['rating'].mean() if not user_ratings.empty else 3.5
    return genre_averages, overall_avg

def estimate_rating_for_new_movie(user_genre_averages, overall_avg, movie_genres, movie_id):
    """Estimate rating for a new movie based on user's genre averages."""
    genres = movie_genres.get(movie_id, [])
    if not genres or genres == ['(no genres listed)']:
        return overall_avg
    genre_ratings = [user_genre_averages.get(genre, overall_avg) for genre in genres]
    return sum(genre_ratings) / len(genre_ratings)

def get_top_new_movies(user_id, ratings_df, movie_genres, new_movie_ids, n=N_NEW_MOVIES):
    """Get top N new movies based on estimated ratings."""
    genre_averages, overall_avg = get_user_genre_averages(user_id, ratings_df, movie_genres)
    estimated_ratings = []
    for movie_id in new_movie_ids:
        est_rating = estimate_rating_for_new_movie(genre_averages, overall_avg, movie_genres, movie_id)
        estimated_ratings.append((movie_id, est_rating))
    return sorted(estimated_ratings, key=lambda x: x[1], reverse=True)[:n]

def get_user_rated_movies(user_id, ratings_df):
    """Get movies rated by the user."""
    if ratings_df is None or user_id not in ratings_df['userId'].values:
        return set(), pd.DataFrame()
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    rated_movie_ids = set(user_ratings['movieId'])
    return rated_movie_ids, user_ratings.sort_values('rating', ascending=False)

def get_existing_user_recommendations(user_id, model, movies_df, ratings_df, raw_to_inner_id_map, n=N_RECOMMENDATIONS):
    """Generate SVD-based recommendations for existing movies."""
    if model is None or movies_df is None or ratings_df is None or not raw_to_inner_id_map:
        st.warning("Cannot get recommendations: Model or data missing.")
        return pd.DataFrame()

    all_movie_ids_in_model = set(raw_to_inner_id_map.keys())
    rated_movie_ids, _ = get_user_rated_movies(user_id, ratings_df)
    unrated_movie_ids = list(all_movie_ids_in_model - rated_movie_ids)

    predictions = []
    for movie_id in unrated_movie_ids:
        pred = model.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))

    top_n_predictions = heapq.nlargest(n, predictions, key=lambda x: x[1])
    top_n_movie_ids = [pred[0] for pred in top_n_predictions]

    recommendations_df = movies_df[movies_df['movieId'].isin(top_n_movie_ids)].copy()
    pred_map = dict(top_n_predictions)
    recommendations_df['predicted_rating'] = recommendations_df['movieId'].map(pred_map)
    recommendations_df = recommendations_df.set_index('movieId').loc[top_n_movie_ids].reset_index()
    return recommendations_df[['movieId', 'title', 'genres', 'genres_list', 'predicted_rating']]

def get_new_user_recommendations(rated_movies_dict, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, movies_df, movie_genres, new_movie_ids, n=N_RECOMMENDATIONS):
    """Generate similarity-based recommendations for new users."""
    if not rated_movies_dict or movies_df is None:
        st.warning("Cannot get new user recommendations: Input or data missing.")
        return pd.DataFrame()

    candidate_scores = {}
    input_movie_ids = set(rated_movies_dict.keys())

    # Calculate scores based on similarity to rated movies
    for movie_id, rating in rated_movies_dict.items():
        if movie_id not in movies_df['movieId'].values:
            st.warning(f"Skipping movie ID {movie_id} (not in dataset).")
            continue
        rating_weight = max(0.1, rating - 2.5)  # Weight by rating deviation from neutral
        is_new_movie = movie_id in new_movie_ids
        similar_movies = get_similar_movies(movie_id, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, movie_genres, is_new_movie=is_new_movie, n=50)
        for sim_movie_id, similarity in similar_movies:
            if sim_movie_id not in input_movie_ids:
                candidate_scores[sim_movie_id] = candidate_scores.get(sim_movie_id, 0) + similarity * rating_weight

    if not candidate_scores:
        st.info("Could not find suitable recommendations based on similarity.")
        return pd.DataFrame()

    # Ensure at least N_NEW_MOVIES new movies are included
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [(mid, score) for mid, score in sorted_candidates if mid not in new_movie_ids][:max(0, n - N_NEW_MOVIES)]
    new_candidates = [(mid, score) for mid, score in sorted_candidates if mid in new_movie_ids][:N_NEW_MOVIES]
    combined_candidates = top_candidates + new_candidates
    if len(combined_candidates) < n:
        additional_needed = n - len(combined_candidates)
        remaining_candidates = [c for c in sorted_candidates if c[0] not in [x[0] for x in combined_candidates]][:additional_needed]
        combined_candidates.extend(remaining_candidates)

    top_n_movie_ids = [movie_id for movie_id, score in combined_candidates[:n]]
    recommendations_df = movies_df[movies_df['movieId'].isin(top_n_movie_ids)].copy()
    score_map = dict(combined_candidates[:n])
    recommendations_df['recommendation_score'] = recommendations_df['movieId'].map(score_map)
    recommendations_df = recommendations_df.sort_values('recommendation_score', ascending=False)
    return recommendations_df[['movieId', 'title', 'genres', 'genres_list', 'recommendation_score']]

def get_genre_counts(movie_ids, movie_genres):
    """Count genres for a list of movie IDs."""
    genre_counts = {}
    for movie_id in movie_ids:
        genres = movie_genres.get(movie_id, [])
        for genre in genres:
            if genre != "(no genres listed)":
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    total = sum(genre_counts.values())
    return {genre: count / total for genre, count in genre_counts.items()} if total > 0 else {}

# --- Plotting Functions ---
def plot_genre_comparison(user_genres, recommended_genres):
    """Plot genre comparison between user ratings and recommendations."""
    genres = set(user_genres.keys()).union(recommended_genres.keys())
    user_vals = [user_genres.get(genre, 0) * 100 for genre in genres]  # Convert to percentage
    rec_vals = [recommended_genres.get(genre, 0) * 100 for genre in genres]
    
    fig = go.Figure(data=[
        go.Bar(name='User Rated', x=list(genres), y=user_vals),
        go.Bar(name='Recommended', x=list(genres), y=rec_vals)
    ])
    fig.update_layout(
        barmode='group',
        title='Genre Distribution: Rated vs Recommended',
        xaxis_title='Genres',
        yaxis_title='Percentage (%)',
        template='plotly_white'
    )
    return fig

def plot_similarity_chart(similarities_list, movie_titles_map):
    """Creates a bar chart showing similarity scores, styled like the screenshot."""
    if not similarities_list:
        st.write("No similarity data to plot.")
        return None
    try:
        df = pd.DataFrame(similarities_list, columns=['movieId', 'similarity'])
        df['title'] = df['movieId'].map(lambda x: movie_titles_map.get(x, f"Unknown Movie (ID: {x})"))
        df = df.sort_values('similarity', ascending=True)  # Ascending for horizontal bar chart

        # Ensure similarity is numeric for plotting
        df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')
        df = df.dropna(subset=['similarity'])
        if df.empty:
            st.write("Similarity data could not be processed for plotting.")
            return None

        # Create the bar chart with styling to match the screenshot
        fig = px.bar(
            df.head(N_SIMILAR_MOVIES_DISPLAY),
            x='similarity',
            y='title',
            orientation='h',
            title=f"Top {min(N_SIMILAR_MOVIES_DISPLAY, len(df))} Rated Movies Driving This Recommendation",
            labels={'similarity': 'Similarity Score', 'title': 'Your Rated Movie'},
            height=300,
            text='similarity'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside') # Format text
        fig.update_layout(yaxis_title="Your Rated Movie", xaxis_title="Similarity Score", yaxis={'categoryorder':'total ascending'})
        return fig
    except Exception as e:
        st.error(f"Error creating similarity chart: {e}")
        return None

def plot_latent_space(pca_df, user_rated_ids, recommended_ids, movie_titles_map, selected_rec_id=None):
    """Plot movies in 2D PCA space."""
    if pca_df is None or pca_df.empty:
        st.write("PCA data not available for plotting.")
        return None
    plot_data = pca_df.copy()
    user_rated_ids_set = set(user_rated_ids)
    recommended_ids_set = set(recommended_ids)
    def get_status(row):
        movie_id = row['movieId']
        if selected_rec_id and movie_id == selected_rec_id:
            return 'Selected Rec'
        elif movie_id in recommended_ids_set:
            return 'Recommended'
        elif movie_id in user_rated_ids_set:
            return 'You Rated'
        else:
            return 'Other Movies'
    plot_data['status'] = plot_data.apply(get_status, axis=1)
    category_orders = ["Other Movies", "You Rated", "Recommended", "Selected Rec"]
    plot_data['status'] = pd.Categorical(plot_data['status'], categories=category_orders, ordered=True)
    plot_data = plot_data.sort_values('status')
    color_map = {"Other Movies": "lightgrey", "You Rated": "blue", "Recommended": "red", "Selected Rec": "yellow"}
    size_map = {"Other Movies": 4, "You Rated": 10, "Recommended": 10, "Selected Rec": 14}
    symbol_map = {"Other Movies": "circle", "You Rated": "circle", "Recommended": "diamond", "Selected Rec": "star"}
    plot_data['size'] = plot_data['status'].map(size_map)
    plot_data['title'] = plot_data['movieId'].map(lambda x: movie_titles_map.get(x, f"Unknown (ID: {x})"))
    fig = px.scatter(plot_data, x='pca_comp_1', y='pca_comp_2', color='status', size='size', symbol='status',
                     color_discrete_map=color_map, symbol_map=symbol_map, hover_name='title',
                     hover_data={'pca_comp_1': False, 'pca_comp_2': False, 'status': True, 'genres': True, 'size': False},
                     title="Movie Latent Space (PCA Visualization)",
                     labels={'pca_comp_1': 'PCA Component 1', 'pca_comp_2': 'PCA Component 2'}, height=500)
    fig.update_layout(legend_title_text='Movie Category', showlegend=True)
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    return fig

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🎬 Visualising Movie Recommendation System")

# Load Data and Train Model
movies_df, ratings_df, movie_ratings_df, movie_titles, movie_genres, new_movie_ids = load_data()
model, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, pca_df = train_svd_model_and_get_factors(ratings_df, movies_df)

# App Initialization Check
APP_READY = all(obj is not None for obj in [movies_df, ratings_df, model, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, pca_df, movie_titles, movie_genres, new_movie_ids])
if not APP_READY:
    st.error("Application initialization failed. Please check logs and data sources.")
    st.stop()

# Initialize session state
if 'selected_recommendation' not in st.session_state:
    st.session_state.selected_recommendation = None
if 'selected_user_for_rec_info' not in st.session_state:
    st.session_state.selected_user_for_rec_info = None
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None

# Main Application Tabs
tab1, tab2 = st.tabs(["👤 Recommend for Existing User", "✨ Recommend for New User"])

with tab1:
    st.header("Select Existing User")
    user_list = sorted(ratings_df['userId'].unique()) if ratings_df is not None else []

    def user_changed():
        st.session_state.selected_recommendation = None
        st.session_state.selected_user_for_rec_info = None
        st.session_state.current_user_id = st.session_state.existing_user_select_widget

    selected_user_id = st.selectbox("Choose a User ID:", user_list, key="existing_user_select_widget", on_change=user_changed)
    if st.session_state.current_user_id is None and selected_user_id:
        st.session_state.current_user_id = selected_user_id
    current_user_id = st.session_state.current_user_id

    if current_user_id:
        st.info(f"Displaying recommendations and analysis for User ID: **{current_user_id}**")
        rated_movie_ids, user_top_ratings = get_user_rated_movies(current_user_id, movie_ratings_df)
        svd_recommendations = get_existing_user_recommendations(current_user_id, model, movies_df, ratings_df, raw_to_inner_id_map, n=N_RECOMMENDATIONS)
        new_movie_recommendations = get_top_new_movies(current_user_id, ratings_df, movie_genres, new_movie_ids, n=N_NEW_MOVIES)

        # Convert new movie recommendations to DataFrame
        new_movie_df = pd.DataFrame(new_movie_recommendations, columns=['movieId', 'estimated_rating'])
        new_movie_df = new_movie_df.merge(movies_df[['movieId', 'title', 'genres', 'genres_list']], on='movieId', how='left')

        recommended_movie_ids = svd_recommendations['movieId'].tolist() if not svd_recommendations.empty else []

        # --- Row 1: User Ratings & Genre Comparison ---
        col1a, col1b = st.columns([1, 2])

        with col1a:
            st.subheader(f"Top Rated by User {current_user_id}")
            if not user_top_ratings.empty:
                for idx, (_, movie) in enumerate(user_top_ratings.head(N_DISPLAY_USER_RATINGS).iterrows()):
                    st.caption(f"**{movie['title']}** ({movie['rating']} ⭐)")
                    st.write(f"_{movie['genres']}_")
                    if idx < N_DISPLAY_USER_RATINGS - 1:
                        st.divider()
            else:
                st.write("No rating history found.")

        with col1b:
            st.subheader("Genre Analysis")
            if not user_top_ratings.empty and not svd_recommendations.empty:
                user_rated_ids_for_genre = user_top_ratings['movieId'].tolist()
                user_rated_genres_perc = get_genre_counts(user_rated_ids_for_genre, movie_genres)
                svd_recommended_genres_perc = get_genre_counts(svd_recommendations['movieId'].tolist(), movie_genres)
                fig_genres = plot_genre_comparison(user_rated_genres_perc, svd_recommended_genres_perc)
                if fig_genres:
                    st.plotly_chart(fig_genres, use_container_width=True)

        st.divider()

        # --- Row 2: SVD Recommendations & New Movie Recommendations ---
        col2a, col2b = st.columns([1, 1])

        with col2a:
            st.subheader(f"Top {N_RECOMMENDATIONS} Recommendations (Existing Movies)")
            if not svd_recommendations.empty:
                st.caption("Click expander ▼ to see details and similarity analysis.")
                for idx, movie in svd_recommendations.iterrows():
                    rec_movie_id = movie['movieId']
                    rec_title = movie['title']
                    rec_genres = movie['genres']
                    rec_pred_rating = movie['predicted_rating']
                    is_selected_for_info = (st.session_state.selected_recommendation == rec_movie_id and
                                            st.session_state.selected_user_for_rec_info == current_user_id)
                    with st.expander(f"**{rec_title}** (Predicted: {rec_pred_rating:.2f} ⭐)", expanded=is_selected_for_info):
                        st.write(f"Genres: _{rec_genres}_")
                        button_key = f"why_button_svd_{rec_movie_id}_{current_user_id}"
                        if st.button("Analyze Similarity", key=button_key):
                            st.session_state.selected_recommendation = rec_movie_id
                            st.session_state.selected_user_for_rec_info = current_user_id
                            st.rerun()
                        if is_selected_for_info:
                            st.markdown("---")
                            st.write(f"**Why might '{rec_title}' be recommended?**")
                            st.write("It's similar to these movies you rated highly (>= 3.5 ⭐):")
                            user_high_rated_movies = user_top_ratings[user_top_ratings['rating'] >= 3.5]['movieId'].tolist()
                            similarities_to_rated = []
                            # Compute similarities directly between the recommended movie and all high-rated movies
                            for rated_id in user_high_rated_movies:
                                if rated_id in raw_to_inner_id_map and rec_movie_id in raw_to_inner_id_map:
                                    rated_inner_id = raw_to_inner_id_map[rated_id]
                                    rec_inner_id = raw_to_inner_id_map[rec_movie_id]
                                    if rated_inner_id < len(item_factors) and rec_inner_id < len(item_factors):
                                        sim = cosine_similarity(item_factors[rec_inner_id], item_factors[rated_inner_id])
                                        similarities_to_rated.append((rated_id, sim))
                            similarities_to_rated.sort(key=lambda x: x[1], reverse=True)
                            fig_sim = plot_similarity_chart(similarities_to_rated, movie_titles)
                            if fig_sim:
                                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.write("Could not generate SVD recommendations.")

        with col2b:
            st.subheader(f"Upcoming Movies You Might Like (New Movies)")
            if not new_movie_df.empty:
                st.caption("Click expander ▼ to see details and similarity analysis.")
                for idx, movie in new_movie_df.iterrows():
                    rec_movie_id = movie['movieId']
                    rec_title = movie['title']
                    rec_genres = movie['genres']
                    rec_est_rating = movie['estimated_rating']
                    is_selected_for_info = (st.session_state.selected_recommendation == rec_movie_id and
                                            st.session_state.selected_user_for_rec_info == current_user_id)
                    with st.expander(f"**{rec_title}** (Estimated: {rec_est_rating:.2f} ⭐)", expanded=is_selected_for_info):
                        st.write(f"Genres: _{rec_genres}_")
                        button_key = f"why_button_new_{rec_movie_id}_{current_user_id}"
                        if st.button("Analyze Similarity", key=button_key):
                            st.session_state.selected_recommendation = rec_movie_id
                            st.session_state.selected_user_for_rec_info = current_user_id
                            st.rerun()
                        if is_selected_for_info:
                            st.markdown("---")
                            st.write(f"**Why might '{rec_title}' be recommended?**")
                            st.write("It's similar to these movies you rated highly (>= 3.5 ⭐):")
                            user_high_rated_movies = user_top_ratings[user_top_ratings['rating'] >= 3.5]['movieId'].tolist()
                            similarities_to_rated = []
                            # For new movies, use genre-based similarity
                            is_new_movie = rec_movie_id in new_movie_ids
                            if is_new_movie:
                                similarities_to_rated = get_similar_movies(rec_movie_id, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, movie_genres, is_new_movie=True, n=N_SIMILAR_MOVIES_DISPLAY)
                                similarities_to_rated = [(mid, sim) for mid, sim in similarities_to_rated if mid in user_high_rated_movies]
                            else:
                                # Compute similarities directly for existing movies
                                for rated_id in user_high_rated_movies:
                                    if rated_id in raw_to_inner_id_map and rec_movie_id in raw_to_inner_id_map:
                                        rated_inner_id = raw_to_inner_id_map[rated_id]
                                        rec_inner_id = raw_to_inner_id_map[rec_movie_id]
                                        if rated_inner_id < len(item_factors) and rec_inner_id < len(item_factors):
                                            sim = cosine_similarity(item_factors[rec_inner_id], item_factors[rated_inner_id])
                                            similarities_to_rated.append((rated_id, sim))
                                similarities_to_rated.sort(key=lambda x: x[1], reverse=True)
                            fig_sim = plot_similarity_chart(similarities_to_rated, movie_titles)
                            if fig_sim:
                                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.write("No new movies available.")

        # --- Row 3: Latent Space Visualization ---
        st.divider()
        st.subheader("Latent Space Visualization")
        if pca_df is not None and not pca_df.empty:
            selected_rec_id_for_plot = st.session_state.selected_recommendation if st.session_state.selected_user_for_rec_info == current_user_id else None
            fig_pca = plot_latent_space(pca_df, rated_movie_ids, recommended_movie_ids, movie_titles, selected_rec_id=selected_rec_id_for_plot)
            if fig_pca:
                st.plotly_chart(fig_pca, use_container_width=True)
                st.caption("Blue = Your Rated, Red = Recommended, Yellow = Selected. Note: New movies (e.g., movieId >= 193610) lack SVD factors and won’t appear here.")
        else:
            st.write("PCA data not available.")

    # --- Tab 2: New User Recommendations ---
# --- Tab 2: New User Recommendations ---
with tab2:
    st.header("Recommend for New User")
    st.write("Let’s find movies you’ll love! First, select the genres you enjoy.")

    # Step 1: Genre Selection (Blocks)
    # Define a list of popular genres (you can adjust this list based on your dataset)
    popular_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Romance', 'Sci-Fi', 'Adventure', 'Fantasy', 'Crime', 'Animation']

    # Initialize session state to store selected genres
    if 'selected_genres' not in st.session_state:
        st.session_state.selected_genres = []

    # Display genres as blocks using columns
    cols_per_row = 5  # Number of genre blocks per row
    for i in range(0, len(popular_genres), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(popular_genres):
                genre = popular_genres[idx]
                # Check if the genre is already selected
                is_selected = genre in st.session_state.selected_genres
                # Use a button with dynamic styling
                button_style = "background-color: #4CAF50; color: white; border: none; padding: 10px; border-radius: 5px;" if is_selected else "background-color: #333333; color: white; border: none; padding: 10px; border-radius: 5px;"
                if cols[j].button(genre, key=f"genre_{genre}", help=f"Click to {'deselect' if is_selected else 'select'} {genre}"):
                    if is_selected:
                        st.session_state.selected_genres.remove(genre)
                    else:
                        st.session_state.selected_genres.append(genre)
                    st.rerun()  # Rerun to update the UI

    # Step 2: Movie Selection (Blocks) - Show only if genres are selected
    if st.session_state.selected_genres:
        st.markdown("---")
        st.write("Great! Now, select the movies you like from these genres.")

        # Initialize session state to store liked movies
        if 'liked_movies' not in st.session_state:
            st.session_state.liked_movies = {}

        # Fetch random movies for the selected genres
        movies_per_genre = 10  # Number of random movies to show per genre
        liked_movies_dict = {}  # To store movie IDs and assumed ratings

        for genre in st.session_state.selected_genres:
            # Filter movies by genre and get random ones
            genre_movies = movie_ratings_df[movie_ratings_df['genres'].str.contains(genre, na=False)]
            if not genre_movies.empty:
                # Group by movieId to avoid duplicates, then sample 10 random movies
                genre_movies = genre_movies[['movieId', 'title', 'genres']].drop_duplicates(subset='movieId')
                if len(genre_movies) > movies_per_genre:
                    genre_movies = genre_movies.sample(n=movies_per_genre, random_state=42)  # Randomly sample 10 movies
                else:
                    genre_movies = genre_movies.head(movies_per_genre)

                # Display movies as blocks
                st.subheader(f"Random {genre} Movies")
                cols_per_row = 3  # Number of movie blocks per row
                for i in range(0, len(genre_movies), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(genre_movies):
                            movie = genre_movies.iloc[idx]
                            movie_id = movie['movieId']
                            movie_title = movie['title']
                            # Check if the movie is already liked
                            is_liked = st.session_state.liked_movies.get(movie_id, False)
                            # Use a button with a unique key incorporating the genre
                            button_label = "Unlike" if is_liked else "Like"
                            button_style = "background-color: #FF4B4B; color: white; border: none; padding: 5px; border-radius: 5px;" if is_liked else "background-color: #1f77b4; color: white; border: none; padding: 5px; border-radius: 5px;"
                            with cols[j]:
                                st.markdown(f"**{movie_title}**")
                                st.markdown(f"_{movie['genres']}_")
                                # Unique key: include genre to avoid duplicates across genres
                                if st.button(button_label, key=f"like_{genre}_{movie_id}", help=f"Click to {'unlike' if is_liked else 'like'} this movie"):
                                    st.session_state.liked_movies[movie_id] = not is_liked
                                    st.rerun()  # Rerun to update the UI

        # Step 3: Generate Recommendations Based on Liked Movies
        if st.session_state.liked_movies:
            # Convert liked movies to a ratings dictionary (assume 4.0 for liked movies)
            liked_movies_dict = {movie_id: 4.0 for movie_id, liked in st.session_state.liked_movies.items() if liked}

            if st.button("Get Recommendations Based on Your Likes", key="new_user_button", type="primary"):
                if not liked_movies_dict:
                    st.warning("Please like at least one movie to get recommendations.")
                else:
                    st.subheader(f"Top {N_RECOMMENDATIONS} Recommendations For You")
                    with st.spinner("Generating recommendations..."):
                        new_recs = get_new_user_recommendations(liked_movies_dict, item_factors, raw_to_inner_id_map, inner_to_raw_id_map, movies_df, movie_genres, new_movie_ids, n=N_RECOMMENDATIONS)
                    
                    if not new_recs.empty:
                        rec_movie_ids_new = new_recs['movieId'].tolist()
                        col_new1, col_new2 = st.columns([1, 1])

                        with col_new1:
                            st.write("**Recommendations:**")
                            for idx, movie in new_recs.iterrows():
                                rec_movie_id = movie['movieId']
                                rec_title = movie['title']
                                rec_genres = movie['genres']
                                rec_score = movie['recommendation_score']
                                with st.expander(f"**{rec_title}** (Score: {rec_score:.2f})"):
                                    st.write(f"Genres: _{rec_genres}_")

                        with col_new2:
                            st.write("**Genre Comparison:**")
                            input_movie_ids = list(liked_movies_dict.keys())
                            input_genres_perc = get_genre_counts(input_movie_ids, movie_genres)
                            recommended_genres_perc_new = get_genre_counts(rec_movie_ids_new, movie_genres)
                            fig_genres_new = plot_genre_comparison(input_genres_perc, recommended_genres_perc_new)
                            if fig_genres_new:
                                fig_genres_new.update_layout(title="Genre Profile: Your Likes vs. Recommendations")
                                st.plotly_chart(fig_genres_new, use_container_width=True)
                    else:
                        st.error("Could not generate recommendations. Try liking different movies.")

# --- Sidebar Info ---
st.sidebar.title("About")
st.sidebar.info("""
    This app uses **SVD collaborative filtering** for existing users and **similarity-based recommendations** for new users,
    with **genre-based estimation** for new movies, **similarity analysis**, and **latent space visualization**.
    - **Existing User**: SVD + genre estimation.
    - **New User**: Similarity-based with SVD factors and genre overlap.
    **Data:** MovieLens (downloaded dynamically) + Future Movies (processed from Top 100 Trending Movies 2025)
    **Model:** Surprise SVD + Scikit-learn PCA
    **Visualization:** Streamlit + Plotly
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Project for Data Visualization Class")