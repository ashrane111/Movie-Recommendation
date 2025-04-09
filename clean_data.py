import pandas as pd
import os

def parse_box_office(value):
    if pd.isna(value):
        return 0  # Handle NaN, though not present in your data
    value = value.replace('$', '').strip()
    if 'billion' in value.lower():
        return float(value.replace(' billion', '')) * 1000
    elif 'million' in value.lower():
        return float(value.replace(' million', ''))
    return 0

def assign_rating(box_office):
    if box_office >= 1000:  # $1 billion or more
        return 4.5  # High rating (4.0–5.0 range)
    elif box_office >= 500:  # $500 million to $1 billion
        return 3.5  # Mid-high rating (3.0–4.0 range)
    else:  # < $500 million
        return 2.5

# --- Configuration ---
# Define the path to your data folder relative to your script location
DATA_FOLDER = 'Data'
MOVIES_FILE = os.path.join(DATA_FOLDER, 'movies.csv')
RATINGS_FILE = os.path.join(DATA_FOLDER, 'ratings.csv')
NEW_MOVIES_FILE = os.path.join(DATA_FOLDER, 'Top_100_Trending_Movies_2025.csv')

# --- Load Data ---
print(f"Loading movies data from: {MOVIES_FILE}")
try:
    movies_df = pd.read_csv(MOVIES_FILE)
    print("Movies data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {MOVIES_FILE} not found. Make sure it's in the '{DATA_FOLDER}' directory.")
    exit() # Exit if file not found

print(f"\nLoading ratings data from: {RATINGS_FILE}")
try:
    ratings_df = pd.read_csv(RATINGS_FILE)
    print("Ratings data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {RATINGS_FILE} not found. Make sure it's in the '{DATA_FOLDER}' directory.")
    exit() # Exit if file not found

# --- Initial Inspection ---
print("\n--- Movies DataFrame ---")
print("First 5 rows:")
print(movies_df.head())
print("\nInfo:")
movies_df.info()
print("\nGenres example:", movies_df['genres'].iloc[0]) # Show how genres are stored

print("\n--- Ratings DataFrame ---")
print("First 5 rows:")
print(ratings_df.head())
print("\nInfo:")
ratings_df.info()
print("\nRating statistics:")
print(ratings_df['rating'].describe())

# --- Basic Merging (Optional but useful) ---
# Merge movie titles into the ratings dataframe for easier analysis later
movie_ratings_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')
movie_ratings_df.drop(columns=['timestamp'], inplace=True) # Drop movieId if not needed
print("\n--- Merged DataFrame (Ratings + Movie Titles/Genres) ---")
print("First 5 rows:")
print(movie_ratings_df.head())
movie_ratings_df.to_csv("Data/movies_ratings.csv", index=False)

### Cleaning the new movies dataset

increament_value = int(193609)
df = pd.read_csv(NEW_MOVIES_FILE)
column_to_check = "Rank"
df = df[df[column_to_check].notna()] 
#df = df.dropna()
df["Rank"] = (df["Rank"].astype(int) + increament_value)

df["Genre"] = df["Genre"].str.replace(",", "|")
df.rename(columns={'Rank': 'movieId', 'Genre': 'genres', 'Title': 'title', "IMDB Rating": "rating"}, inplace=True)
df['box_office_millions'] = df['Box Office Prediction'].apply(parse_box_office)
df['rating'] = df['rating'].fillna(
    df['box_office_millions'].apply(assign_rating)
)
df.drop(columns=['Release Date', 'Unnamed: 15' ,'Duration', 'Synopsis', 'Age Rating', 'Streaming Platform', 'Main Cast', 'Box Office Prediction', 'box_office_millions', 'Director', 'Production Company', 'Country', 'Language'], inplace=True)
df.to_csv("Data/updated_new_movies.csv", index=False)