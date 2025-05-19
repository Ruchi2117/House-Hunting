import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

def download_kaggle_dataset():
    try:
        print("Downloading dataset from Kaggle...")
        # Load the dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "yasserh/housing-prices-dataset",
            "Housing.csv"  # The main dataset file
        )
        
        # Save to CSV
        output_path = "data/house_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        # Print dataset info
        print("\nDataset Info:")
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_kaggle_dataset()
