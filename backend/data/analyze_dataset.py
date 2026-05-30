"""Quick EDA on the merged dataset before training."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent / "autism_merged.csv"
df = pd.read_csv(DATA)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)

print(f"Shape: {df.shape}\n")
print("Missing values per column:")
print(df.isna().sum()[df.isna().sum() > 0], "\n")

print("Target distribution (Class/ASD):")
print(df["Class/ASD"].value_counts(dropna=False), "\n")

print("Target by age_group:")
print(pd.crosstab(df["age_group"], df["Class/ASD"]), "\n")

print("Age stats by group:")
print(df.groupby("age_group")["age"].describe()[["count", "mean", "min", "max"]], "\n")

print("gender values:", df["gender"].unique())
print("jaundice values:", df["jaundice"].unique())
print("austim values:", df["austim"].unique(), "\n")

# AQ-10 sum vs label (illustrates the label-derivation / leakage concern)
aq_cols = [f"A{i}_Score" for i in range(1, 11)]
df["aq_sum"] = df[aq_cols].sum(axis=1)
print("Mean AQ-10 sum by class:")
print(df.groupby("Class/ASD")["aq_sum"].mean(), "\n")

print("AQ sum >= 6 vs Class/ASD (the classic AQ-10 referral threshold):")
df["aq_ge6"] = (df["aq_sum"] >= 6).astype(int)
df["label"] = (df["Class/ASD"] == "YES").astype(int)
print(pd.crosstab(df["aq_ge6"], df["label"]))
agree = (df["aq_ge6"] == df["label"]).mean()
print(f"\n'AQ>=6' rule agrees with stored label {agree*100:.1f}% of the time.")
