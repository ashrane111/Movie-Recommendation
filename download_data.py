# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd

# Create Data folder if it doesn't exist
data_folder = "Data"
os.makedirs(data_folder, exist_ok=True)

try:
    # Set the path to the file you'd like to load
    file_path_1 = "movies.csv"  
    file_path_2 = "ratings.csv"
    
    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "shubhammehta21/movie-lens-small-latest-dataset",
        file_path_1,
    )
    
    # Save the dataframe to CSV in the Data folder
    output_path = os.path.join(data_folder, file_path_1)
    df.to_csv(output_path, index=False)
    
    print("First 5 records:")
    print(df.head())
    print(f"\nData successfully saved to: {output_path}")

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "shubhammehta21/movie-lens-small-latest-dataset",
        file_path_2,
    )
    
    # Save the dataframe to CSV in the Data folder
    output_path = os.path.join(data_folder, file_path_2)
    df.to_csv(output_path, index=False)
    
    print("First 5 records:")
    print(df.head())
    print(f"\nData successfully saved to: {output_path}")

    file_path = "Top_100_Trending_Movies_2025.csv"  
    
    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "taimoor888/top-100-trending-movies-of-2025",
        file_path,
    )
    
    # Save the dataframe to CSV in the Data folder
    output_path = os.path.join(data_folder, file_path)
    df.to_csv(output_path, index=False)
    
    print("First 5 records:")
    print(df.head())
    print(f"\nData successfully saved to: {output_path}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your internet connection, Kaggle credentials, or dataset availability.")


