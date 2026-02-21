import numpy as np
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def read_data():
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
   
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    

    # return train_val_object, x_test_df
    return x_train_df, y_train_df, x_test_df

def clean_text(text):
  # keeps only alphanumeric chars and spaces
  return re.sub(r'[^a-zA-Z0-9 ]', '', text)

def build_model():
    preprocessor = ColumnTransformer(transformers=[
        ('tfidf', TfidfVectorizer(preprocessor=clean_text), 'text'),
    #     ('numeric', StandardScaler(), ['word_count', 'avg_word_length', 'sentence_count', 'avg_sentence_length']),
    # ])
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression()) 
    ])

    return pipeline

def hyperparam_tuning(X, y):
    # input: training and test data
    # output: hyperparams that perform best on cross fold validation
    # call build model on grid search hparams
    model = build_model()
    param_grid = {
        'classifier__C': np.logspace(-3, 3, 7),
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__class_weight': [None, 'balanced'],

        # access TfidfVectorizer inside ColumnTransformer: preprocessor__tfidf__*
        'preprocessor__tfidf__max_df': [0.8, 0.9, 1.0],
        'preprocessor__tfidf__min_df': [0.01, 0.03, 1, 2, 5],
        'preprocessor__tfidf__stop_words': ['english', None]
    }
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        verbose=1,
        random_state=42,
        scoring='roc_auc',
        n_jobs=-1
    )

    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def generate_final_test_results(pipeline, _x_test_df):
    # return predictions on test set
    # x_test_text = _x_test_df['text'].values
    # predictions = pipeline.predict(x_test_text)
    # prediction_probs = pipeline.predict_proba(x_test_text)
       # pipeline expects the DataFrame with a 'text' column (ColumnTransformer)
    predictions = pipeline.predict(_x_test_df)
    prediction_probs = pipeline.predict_proba(_x_test_df)
    
    return predictions, prediction_probs

if __name__ == '__main__':
    x_train_df, y_train_df, x_test_df = read_data()
    
    # Extract text data and labels
    # X_train = x_train_df['text'].values
    # Provide the DataFrame so the ColumnTransformer can select the 'text' column
    X_train = x_train_df
    y_train = y_train_df['Coarse Label'].values.ravel()  # Extract 1D array from DataFrame
    


    # Perform hyperparameter tuning
    best_model = hyperparam_tuning(X_train, y_train)
    
    # Generate predictions on test set
    test_predictions, test_probs = generate_final_test_results(best_model, x_test_df)
    
    print(f"\nTest set predictions shape: {test_predictions.shape}")
    print(f"Test set probabilities shape: {test_probs.shape}")

    # extract probability of class 1 (positive class)
    y_proba_test = test_probs[:, 1]
    np.savetxt('yproba1_test.txt', y_proba_test, fmt='%.6f')
