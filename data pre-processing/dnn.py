#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


df = pd.read_csv("final_airbnb_dataset.csv")

df.columns.tolist()



# In[29]:


features = [
    'distance_score', 'price_score', 'number_of_reviews',
    'availability_365', 'review_scores_rating',
]

df_model = df.dropna(subset=features + ['booked'])

X = df_model[features]
y = df_model['booked'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)


# In[ ]:





# In[31]:


class AirbnbModel(nn.Module):
    def __init__(self, input_dim):
        super(AirbnbModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer
model = AirbnbModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[43]:


num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Predict probabilities on the full dataset
# Make predictions
with torch.no_grad():
    all_features_tensor = torch.tensor(df_model[features].values, dtype=torch.float32)
    probs = model(all_features_tensor).numpy().flatten()

# Ensure the shapes match
assert len(probs) == len(df_model), "Shape mismatch: probs doesn't align with df_model"

# Assign predictions
df_model = df_model.copy()
df_model['predicted_proba'] = probs

# Confirm it worked
print(df_model.columns)
print(df_model[['predicted_proba']].head())


# In[33]:


print(df_model.columns.tolist())


# In[44]:


top_10 = df_model.sort_values(by='predicted_proba', ascending=False).head(10)
top_10_links = top_10[['id', 'listing_url', 'predicted_proba']]

print("\n🔗 Top 10 Recommended Airbnbs with URLs:\n")
for _, row in top_10_links.iterrows():
    print(f"ID: {row['id']} | Score: {row['predicted_proba']:.2f} | URL: {row['listing_url']}")


# In[45]:


plt.figure(figsize=(12, 6))
sorted_probs = df_model['predicted_proba'].sort_values(ascending=False).reset_index(drop=True)
plt.plot(sorted_probs, marker='o', linestyle='', alpha=0.6)
plt.axhline(top_10['predicted_proba'].min(), color='red', linestyle='--', label="Top 10 Cutoff")
plt.title('Sorted Predicted Probabilities')
plt.xlabel('Listing Order (Most to Least Likely to be Booked)')
plt.ylabel('Probability of Booking')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

