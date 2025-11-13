import pandas as pd
from pathlib import Path

src = Path("data/data.csv")
dst = Path("data/new_data.csv")

def main():
    df = pd.read_csv(src, sep=';')

    first_col = df.columns[0]

    df[["Sentence", "Sentiment"]] = df[first_col].str.rsplit(",", n=1, expand=True)

    df = df[["Sentence", "Sentiment"]].dropna()

    print(df.head())
    print(df["Sentiment"].value_counts())

    dst.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(dst, index=False)
    print(f"\nSaved file in: {dst}")

if __name__ == "__main__":
    main()
