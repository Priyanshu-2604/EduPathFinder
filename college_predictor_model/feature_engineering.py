import pandas as pd

# Load cleaned data
df = pd.read_csv('cleaned_iit_ranks.csv')

# Example: Add feature for rank range
df['rank_range'] = df['closing_rank'] - df['opening_rank']

# Example: Analyze program popularity
top_programs = df['academic_program_name'].value_counts().head(10)
print('Top 10 programs by frequency:')
print(top_programs)

# Save engineered features
df.to_csv('features_iit_ranks.csv', index=False)
print('Feature-engineered data saved to features_iit_ranks.csv')