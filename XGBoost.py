import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt

def run_xgboost_model(event_name):
    # Load the dataset
    df = pd.read_csv("final_airbnb_dataset.csv")

    # Define features and target variable
    features = ['distance_score', 'price_score', 'number_of_reviews', 'availability_365', 'review_scores_rating']
    df_model = df.dropna(subset=features + ['booked'])

    X = df_model[features]
    y = df_model['booked'].astype(int)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3. Set up XGBoost parameters
    # For classification (if your labels are discrete classes)
    params_clf = {
        'objective': 'binary:logistic',  # or 'multi:softmax' for multi-class
        'eval_metric': 'logloss',        # or 'mlogloss' for multi-class
        'eta': 0.1,                      # learning rate
        'max_depth': 6,                  # tree depth
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # 4. Choose which parameter set to use based on your problem
    params = params_clf

    # 5. Convert data to DMatrix format (XGBoost's optimized data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 6. Train the model with early stopping
    num_rounds = 1000
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evallist,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # 7. Make predictions
    y_pred = model.predict(dtest)

    # For classification, convert probabilities to classes if needed

    if params['objective'] == 'binary:logistic':
        y_pred_class = [1 if p >= 0.5 else 0 for p in y_pred]
    else:  # multi-class
        y_pred_class = np.argmax(y_pred, axis=1)



    # 8. Feature importance analysis
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    # plt.show()

    # save the plot
    plt.savefig("xgboost_feature_importance.png")
    plt.close()

    # print out top 10 airbnb listings:
    top_10_indices = np.argsort(y_pred)[-10:][::-1]
    top_10_listings = df.iloc[top_10_indices]

    # write to a new .txt file:
    with open("xgboost_recommendations.txt", "w") as f:
        f.write(f"Hey there! Based on your favorite artist, check out these Airbnbs for their upcoming concert: {event_name}\n\n")
        f.write("XGBoost: Top 10 Recommended Airbnbs:\n")
        for _, row in top_10_listings.iterrows():
            f.write(f"ID: {row['id']}, Score: {y_pred[row.name]:.4f}\n")
        # Accuracy metrics: 
        f.write(f"\nAccuracy: {accuracy_score(y_test, y_pred_class):.4f}\n")
        f.write(f"Precision: {precision_score(y_test, y_pred_class, average='weighted'):.4f}\n")
        f.write(f"Recall: {recall_score(y_test, y_pred_class, average='weighted'):.4f}\n")
        f.write(f"F1 Score: {f1_score(y_test, y_pred_class, average='weighted'):.4f}\n")

    print("✅ XGBoost: Top 10 Recommended Airbnbs saved to xgboost_recommendations.txt")

    # 9. Save the model
    model.save_model('recommendation_xgboost_model.json')
        

