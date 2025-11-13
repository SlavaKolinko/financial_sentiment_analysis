import pandas as pd
from pathlib import Path

def main():
    src = Path("data/new_data.csv")
    df = pd.read_csv(src)
    print("Head:")
    print(df.head())
    print("\nLabel distribution:")
    print(df['Sentiment'].value_counts())

if __name__ == "__main__":
    main()
