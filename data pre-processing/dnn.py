import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import os
from sklearn.metrics import accuracy_score



df = pd.read_csv("final_airbnb_dataset.csv")

df.columns.tolist()

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


def run_dnn_model(event_name):
    df = pd.read_csv("final_airbnb_dataset.csv")
    features = ['distance_score', 'price_score', 'number_of_reviews', 'availability_365', 'review_scores_rating']
    df_model = df.dropna(subset=features + ['booked'])

    X = df_model[features]
    y = df_model['booked'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    model = AirbnbModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(800):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_probs = model(X_test_tensor).numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, test_preds)
    print(f"\n✅ DNN Model Accuracy on Test Set: {accuracy:.4f}")
    # Predictions
    with torch.no_grad():
        all_features_tensor = torch.tensor(df_model[features].values, dtype=torch.float32)
        probs = model(all_features_tensor).numpy().flatten()

    df_model = df_model.copy()
    df_model['predicted_proba'] = probs

    top_10 = df_model.sort_values(by='predicted_proba', ascending=False).head(10)
    top_10['listing_url'] = top_10['listing_url'].apply(
        lambda url: f'<a href="{url}" target="_blank">{url}</a>'
    )
    top_10_links = top_10[['id', 'listing_url', 'predicted_proba']]

    # Visualization
    #plt.figure(figsize=(12, 6))
    #sorted_probs = df_model['predicted_proba'].sort_values(ascending=False).reset_index(drop=True)
    #plt.plot(sorted_probs, marker='o', linestyle='', alpha=0.6)
    #plt.axhline(top_10['predicted_proba'].min(), color='red', linestyle='--', label="Top 10 Cutoff")
    #plt.title('Sorted Predicted Probabilities')
    #plt.xlabel('Listing Order (Most to Least Likely to be Booked)')
    #plt.ylabel('Probability of Booking')
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.savefig("dnn_prediction_plot.png")
    #plt.close()


    # Save and open HTML report
    heading_html = f"<h2>Hey there! Based on your favorite artist, check out these Airbnbs for their upcoming concert: <strong>{event_name}</strong></h2>"
    table_html = top_10_links.to_html(index=False, escape=False)

    output_file = os.path.abspath("dnn_report.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"""
        <html>
            <head>
                <title>DNN Predictions</title>
            </head>
            <body>
                {heading_html}
                <br>
                {table_html}
            </body>
        </html>
        """)

    print(f"\n✅ DNN Predictions saved to: {output_file}")
    return output_file  # So main.py can open this


#if __name__ == "__main__":
#    run_dnn_model(event_name)
