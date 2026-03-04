"""Train and evaluate a linear SVM classifier for reading level.

This script uses :mod:`preprocessing` to load and preprocess the data, then
fits a linear support vector machine to predict the reading level (fine-label)
from the features (including BERT embeddings).  A simple train/validation split
is used for evaluation, and predictions on the held-out test set are written to
CSV.
"""

import os
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from preprocessing import preprocess


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42):
    """Load data, train a linear SVM, and evaluate its performance.

    Parameters
    ----------
    test_size : float
        Fraction of the training data to hold out for validation.
    random_state : int
        Seed for reproducible randomness.
    """
    x_train, y_train, x_test = preprocess()

    label_col = 'Coarse Label'

    y = y_train[label_col].astype(str)
    groups = y_train['author']

    # encode string labels as integers for the classifier
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # set up parameter grid and cross-validation
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [2, 5,6], 'gamma': ['scale', 'auto']},
    ]

    k_folds = 3
    cv = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=SVC(probability=True),
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        refit=True
    )

    print(f"Starting GridSearchCV ({k_folds}-fold) over {len(param_grid)} parameter sets...")
    grid.fit(x_train, y_enc, groups=groups)

    print("Best CV score:", grid.best_score_)
    print("Best parameters:", grid.best_params_)

    best_clf = grid.best_estimator_

    # generate test predictions using the best estimator (already refit on full train)
    y_test_pred = best_clf.predict(x_test)
    y_test_labels = le.inverse_transform(y_test_pred)
    
    # get probability predictions
    y_test_proba = best_clf.predict_proba(x_test)
    # extract probability of Class 1 (positive class: UK Key Stage 4-5)
    y_test_proba_class1 = y_test_proba[:, 1]

    output_path = os.path.join(os.getcwd(), 'linear_svm_predictions.csv')
    pd.DataFrame({label_col: y_test_labels}).to_csv(output_path, index=False)
    print(f"Test set predictions saved to {output_path}")
    
    # save probability predictions to text file (one float per line)
    proba_output_path = os.path.join(os.getcwd(), 'yproba1_test.txt')
    with open(proba_output_path, 'w') as f:
        for prob in y_test_proba_class1:
            f.write(f"{prob}\n")
    print(f"Test set probability predictions saved to {proba_output_path}")


if __name__ == '__main__':
    train_and_evaluate()
