import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from preprocessing import preprocess
import matplotlib.pyplot as plt

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

x_train, y_train_df, x_test = preprocess()
y_train = (y_train_df['Coarse Label'].values.ravel() == 'Key Stage 4-5').astype(int)


# ── 2. DEFINE PIPELINE ────────────────────────────────────────────────────────

# Note: XGBoost is tree-based so scaling doesn't affect it,
# but we keep StandardScaler for consistency with knn.py
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
])

param_dist = {
    'xgb__n_estimators':  [100, 200, 300, 500],       # number of trees
    'xgb__max_depth':     [3, 4, 5, 6, 8],            # tree depth — higher = more complex
    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],     # shrinkage per tree
    'xgb__subsample':     [0.6, 0.8, 1.0],            # fraction of rows per tree
    'xgb__colsample_bytree': [0.6, 0.8, 1.0]          # fraction of features per tree
}


# ── 3. RANDOMIZED SEARCH ──────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(x_train, y_train)

print(f"\nBest params:      {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.4f}")


# ── 4. EVALUATE BEST MODEL ────────────────────────────────────────────────────

best_model = search.best_estimator_
yhat_train = best_model.predict(x_train)

p = search.best_params_
cm = confusion_matrix(y_train, yhat_train)
disp = ConfusionMatrixDisplay(cm, display_labels=['low', 'high'])
disp.plot(cmap='Blues')
plt.title(f"Train Confusion Matrix\n"
          f"n_estimators={p['xgb__n_estimators']}, "
          f"max_depth={p['xgb__max_depth']}, "
          f"lr={p['xgb__learning_rate']}")
plt.show()


# ── 5. PREDICT ON TEST SET ────────────────────────────────────────────────────

y_test_pred = best_model.predict(x_test)

# RESULTS
# Fitting 3 folds for each of 20 candidates, totalling 60 fits
# Best params:      {'xgb__subsample': 0.6, 'xgb__n_estimators': 300, 'xgb__max_depth': 8, 'xgb__learning_rate': 0.01, 'xgb__colsample_bytree': 0.6}
# Best CV accuracy: 0.7267
# Conufsion matrix in Latex doc