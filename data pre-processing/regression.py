import pandas as pd

# Load datasets
original = pd.read_csv("listings.csv")
cleaned = pd.read_csv("cleaned_data_nprices.csv")

# Select important columns from original
important_cols = [
    'id', 'latitude', 'longitude', 'neighbourhood', 'neighbourhood_cleansed',
    'property_type', 'room_type', 'accommodates', 'bedrooms', 'bathrooms',
    'number_of_reviews', 'review_scores_rating'
]

original_reduced = original[important_cols]

# Merge with cleaned dataset
merged = cleaned.merge(original_reduced, on='id', how='left')

# Save final dataset
merged.to_csv("final_airbnb_dataset.csv", index=False)

print("Final shape:", merged.shape)
print("Columns in final dataset:", merged.columns)
