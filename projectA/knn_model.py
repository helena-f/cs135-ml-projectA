import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from preprocessing import preprocess
import matplotlib.pyplot as plt

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

x_train, y_train_df, x_test = preprocess()
y_train = (y_train_df['Coarse Label'].values.ravel() == 'Key Stage 4-5').astype(int)


# ── 2. DEFINE PIPELINE ────────────────────────────────────────────────────────

# Pipeline chains steps so PCA + KNN are tuned together,
# and scaling/PCA are correctly fit only on training folds
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

param_dist = {
    'pca__n_components': [20, 50, 100, 150, 200],
    'knn__n_neighbors':  list(range(1, 100, 2)),
    'knn__metric':       ['euclidean', 'cosine'],
    'knn__weights':      ['uniform', 'distance']
}


# ── 3. RANDOMIZED SEARCH ──────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,              # number of random combinations to try
    cv=cv,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1               # use all available cores
)

search.fit(x_train, y_train)

print(f"\nBest params:   {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.4f}")


# ── 4. EVALUATE BEST MODEL ────────────────────────────────────────────────────

best_model = search.best_estimator_
yhat_train = best_model.predict(x_train)

cm = confusion_matrix(y_train, yhat_train)
disp = ConfusionMatrixDisplay(cm, display_labels=['low', 'high'])
disp.plot(cmap='Blues')
plt.title(f"Train Confusion Matrix\n"
          f"k={search.best_params_['knn__n_neighbors']}, "
          f"pca={search.best_params_['pca__n_components']}, "
          f"metric={search.best_params_['knn__metric']}, "
          f"weights={search.best_params_['knn__weights']}")
plt.show()


# ── 5. PREDICT ON TEST SET ────────────────────────────────────────────────────

y_test_pred = best_model.predict(x_test)

# Results
# Fitting 5 folds for each of 50 candidates, totalling 250 fits
# Best params:   {'pca__n_components': 100, 'knn__weights': 'uniform', 'knn__n_neighbors': 81, 'knn__metric': 'cosine'}
# Best CV accuracy: 0.7159
# Conufsion matrix in Latex doc