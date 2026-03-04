import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import RandomizedSearchCV, validation_curve
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

def plot_hyperparameter_complexity(pipeline, X, y, param_name='classifier__C',
                                    param_values=None, cv=5, metric='roc_auc',
                                    title='Model Performance vs Regularization Parameter C',
                                    save_path='hyperparameter_complexity.png'):
    """
    Plot model performance across different hyperparameter values to show
    underfitting, optimal, and overfitting regimes.
    
    Parameters:
    -----------
    pipeline : sklearn Pipeline
        The model pipeline to evaluate.
    X : DataFrame
        Training features.
    y : array-like
        Training labels.
    param_name : str
        Name of the parameter to vary (default: 'classifier__C').
    param_values : array-like, optional
        Values of the hyperparameter to test. If None, uses logspace(-3, 3, 7).
    cv : int
        Number of cross-validation folds (default: 5).
    metric : str
        Scoring metric (default: 'roc_auc').
    title : str
        Title for the figure.
    save_path : str
        Path to save the figure.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : matplotlib.axes.Axes
        The axes object.
    """
    if param_values is None:
        param_values = np.logspace(-3, 3, 7)
    
    # Use validation_curve to get train and validation scores across CV folds
    train_scores, val_scores = validation_curve(
        pipeline, X, y, param_name=param_name, param_range=param_values,
        cv=cv, scoring=metric, n_jobs=-1
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot individual fold points (raw data points for each CV fold)
    # Training folds in blue
    for fold in range(train_scores.shape[1]):
        ax.scatter(param_values, train_scores[:, fold], 
                  alpha=0.35, s=50, color='blue', marker='o')
    
    # Validation folds in orange
    for fold in range(val_scores.shape[1]):
        ax.scatter(param_values, val_scores[:, fold], 
                  alpha=0.35, s=50, color='orange', marker='s')
    
    # Plot mean lines and standard deviation bands
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Mean lines
    ax.plot(param_values, train_mean, 'o-', color='blue', label='Training (mean)', 
            linewidth=2.5, markersize=8)
    ax.plot(param_values, val_mean, 's-', color='orange', label='Validation (mean)', 
            linewidth=2.5, markersize=8)
    
    # Standard deviation bands
    ax.fill_between(param_values, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color='blue', label='Training ±1 std')
    ax.fill_between(param_values, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color='orange', label='Validation ±1 std')
    
    # Formatting
    ax.set_xlabel('Hyperparameter C (Inverse Regularization Strength)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add caption
    caption = (f"Figure shows {metric} performance across {cv}-fold cross-validation.\n"
               f"Small points represent individual CV fold results; solid lines show fold averages.\n"
               f"Shaded regions indicate ±1 standard deviation across folds.\n"
               f"Lower C values increase regularization (underfitting risk), higher C values decrease it (overfitting risk).")
    fig.text(0.1, -0.08, caption, ha='left', fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    return fig, ax

if __name__ == '__main__':
    x_train_df, y_train_df, x_test_df = read_data()
    
    # Extract text data and labels
    # X_train = x_train_df['text'].values
    # Provide the DataFrame so the ColumnTransformer can select the 'text' column
    X_train = x_train_df
    y_train = y_train_df['Coarse Label'].values.ravel()  # Extract 1D array from DataFrame
    


    # Perform hyperparameter tuning
    best_model = hyperparam_tuning(X_train, y_train)
    
    # Plot hyperparameter complexity curve for C parameter
    C_values = np.logspace(-2, 2, num=10)
    plot_hyperparameter_complexity(
        build_model(), X_train, y_train,
        param_name='classifier__C',
        param_values=C_values,
        cv=5,
        metric='roc_auc',
        title='Model Complexity vs Performance: Regularization Parameter C',
        save_path='hyperparameter_C_complexity.png'
    )
    
    # Generate predictions on test set
    test_predictions, test_probs = generate_final_test_results(best_model, x_test_df)
    
    print(f"\nTest set predictions shape: {test_predictions.shape}")
    print(f"Test set probabilities shape: {test_probs.shape}")

    # extract probability of class 1 (positive class)
    y_proba_test = test_probs[:, 1]
    np.savetxt('yproba1_test.txt', y_proba_test, fmt='%.6f')
