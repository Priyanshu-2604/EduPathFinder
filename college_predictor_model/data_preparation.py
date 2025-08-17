import pandas as pd

# Load combined CSV data
csv_path = '../data/extracted/iit_ranks_recent_years.csv'
df = pd.read_csv(csv_path)

# Basic preprocessing: ensure correct types and clean missing values
df['opening_rank'] = pd.to_numeric(df['opening_rank'], errors='coerce')
df['closing_rank'] = pd.to_numeric(df['closing_rank'], errors='coerce')
df = df.dropna(subset=['opening_rank', 'closing_rank'])

# Standardize category and gender labels
df['seat_type'] = df['seat_type'].str.upper().str.strip()
df['gender'] = df['gender'].str.title().str.strip()

# Save cleaned data for modeling
cleaned_path = 'cleaned_iit_ranks.csv'
df.to_csv(cleaned_path, index=False)

print(f'Cleaned data saved to {cleaned_path}. Shape: {df.shape}')