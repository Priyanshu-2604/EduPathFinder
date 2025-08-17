import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Load feature-engineered data
df = pd.read_csv('features_iit_ranks.csv')

# For demonstration, let's predict whether a student with a given rank, category, and gender can get a particular program (binary classification)
# Create a binary target: 1 if rank falls within opening-closing range, else 0
# This is a simplification; for real deployment, use multi-label or recommendation approaches

def eligibility(row, rank, seat_type, gender):
    return (
        row['seat_type'] == seat_type and
        row['gender'] == gender and
        row['opening_rank'] <= rank <= row['closing_rank']
    )

# Example: Simulate for rank/category/gender
rank = 5000
seat_type = 'OPEN'
gender = 'Gender-Neutral'
df['eligible'] = df.apply(lambda row: eligibility(row, rank, seat_type, gender), axis=1)

# Features for modeling
feature_cols = ['year', 'opening_rank', 'closing_rank', 'rank_range']
X = df[feature_cols]
y = df['eligible'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')

# Save model (optional)
import joblib
joblib.dump(model, 'college_predictor_rf.joblib')
print('Model saved to college_predictor_rf.joblib')