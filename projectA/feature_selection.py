import numpy as np
import pandas as pd
import os
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def read_data():
    data_dir = 'data_readinglevel'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    
    return x_train_df, y_train_df, x_test_df


# Define feature groups — keep only the requested five numeric features.
# We preserve the TF-IDF ('text') pipeline for bag-of-words functionality.
BASIC_FEATURES = ['sentence_count']

LINGUISTIC_FEATURES = ['avg_sentence_length', 'punctuation_frequency']

SENTIMENT_FEATURES = []

READABILITY_FEATURES = ['readability_ARI', 'readability_Kincaid']

INFO_FEATURES = []

# Consolidated list of allowed numeric features
ALL_NUMERIC_FEATURES = (BASIC_FEATURES + LINGUISTIC_FEATURES + 
                        SENTIMENT_FEATURES + READABILITY_FEATURES + INFO_FEATURES)


def build_pipeline(numeric_features):
    """Build a pipeline with TF-IDF + specified numeric features"""
    
    transformers = [('tfidf', TfidfVectorizer(max_features=1000), 'text')]
    
    if numeric_features:
        transformers.append(('numeric', StandardScaler(), numeric_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=200, random_state=42))
    ])
    
    return pipeline


def evaluate_feature_combination(x_train_df, y_train, numeric_features, cv=5):
    """Evaluate a feature combination using cross-validation"""
    
    try:
        pipeline = build_pipeline(numeric_features)
        scores = cross_val_score(pipeline, x_train_df, y_train, 
                               cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean(), scores.std()
    except Exception as e:
        print(f"Error evaluating {numeric_features}: {e}")
        return 0, 0


def test_individual_feature_groups(x_train_df, y_train):
    """Test each feature group individually"""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL FEATURE GROUPS")
    print("="*80)
    
    results = []
    
    # Test text only
    score, std = evaluate_feature_combination(x_train_df, y_train, [])
    results.append(("Text Only", [], score, std))
    print(f"Text Only: {score:.4f} (+/- {std:.4f})")
    
    # Test each group
    feature_groups = {
        "Basic Features": BASIC_FEATURES,
        "Linguistic Features": LINGUISTIC_FEATURES,
        "Sentiment Features": SENTIMENT_FEATURES,
        "Readability Features": READABILITY_FEATURES,
        "Info Features": INFO_FEATURES,
    }
    
    for group_name, features in feature_groups.items():
        score, std = evaluate_feature_combination(x_train_df, y_train, features)
        results.append((group_name, features, score, std))
        print(f"{group_name}: {score:.4f} (+/- {std:.4f})")
    
    return results


def test_combined_feature_groups(x_train_df, y_train):
    """Test combinations of feature groups"""
    print("\n" + "="*80)
    print("TESTING COMBINATIONS OF FEATURE GROUPS")
    print("="*80)
    
    results = []
    feature_groups = {
        "Basic": BASIC_FEATURES,
        "Linguistic": LINGUISTIC_FEATURES,
        "Sentiment": SENTIMENT_FEATURES,
        "Readability": READABILITY_FEATURES,
        "Info": INFO_FEATURES,
    }
    
    # Test all pairs of groups
    group_names = list(feature_groups.keys())
    for r in range(1, len(group_names) + 1):
        for combo in combinations(group_names, r):
            combined_features = []
            for group_name in combo:
                combined_features.extend(feature_groups[group_name])
            
            combo_name = " + ".join(combo)
            score, std = evaluate_feature_combination(x_train_df, y_train, combined_features)
            results.append((combo_name, combined_features, score, std))
            print(f"{combo_name}: {score:.4f} (+/- {std:.4f})")
    
    return results


def test_top_individual_features(x_train_df, y_train):
    """Test each feature individually to identify best ones"""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL FEATURES (Top 15)")
    print("="*80)
    
    results = []
    
    for feature in ALL_NUMERIC_FEATURES:
        score, std = evaluate_feature_combination(x_train_df, y_train, [feature])
        results.append((feature, score, std))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 15
    for i, (feature, score, std) in enumerate(results[:15], 1):
        print(f"{i:2d}. {feature:40s}: {score:.4f} (+/- {std:.4f})")
    
    return results


def test_best_individual_features_combinations(x_train_df, y_train, top_n=10):
    """Combine the top individual features"""
    print("\n" + "="*80)
    print(f"TESTING COMBINATIONS OF TOP {top_n} INDIVIDUAL FEATURES")
    print("="*80)
    
    # First get top features
    individual_scores = []
    for feature in ALL_NUMERIC_FEATURES:
        score, std = evaluate_feature_combination(x_train_df, y_train, [feature])
        individual_scores.append((feature, score, std))
    
    individual_scores.sort(key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in individual_scores[:top_n]]
    
    print(f"Top {top_n} features: {top_features[:5]}... (showing first 5)")
    
    results = []
    
    # Test combinations of top features (all subsets)
    for r in range(1, min(6, top_n + 1)):  # Test subsets of size 1-5
        for combo in combinations(top_features, r):
            combo_features = list(combo)
            score, std = evaluate_feature_combination(x_train_df, y_train, combo_features)
            results.append((combo_features, score, std))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10 combinations
    print(f"\nTop combinations:")
    for i, (features, score, std) in enumerate(results[:10], 1):
        features_str = " + ".join(features)
        print(f"{i:2d}. {features_str}: {score:.4f} (+/- {std:.4f})")
    
    if results:
        best_features, best_score, best_std = results[0]
        return best_features, best_score, best_std
    
    return [], 0, 0


def main():
    print("Loading data...")
    x_train_df, y_train_df, x_test_df = read_data()
    y_train = y_train_df['Coarse Label'].values.ravel()
    
    print(f"Training set shape: {x_train_df.shape}")
    print(f"Available features: {len(x_train_df.columns)}")
    
    # Run feature selection tests
    individual_results = test_individual_feature_groups(x_train_df, y_train)
    combined_results = test_combined_feature_groups(x_train_df, y_train)
    top_features_results = test_top_individual_features(x_train_df, y_train)
    best_combo_features, best_score, best_std = test_best_individual_features_combinations(
        x_train_df, y_train, top_n=10
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: BEST CONFIGURATIONS")
    print("="*80)
    
    # Find best text + feature combination
    all_group_results = individual_results + combined_results
    best_combined = max(all_group_results, key=lambda x: x[2])
    print(f"\nBest feature group combination:")
    print(f"  Config: {best_combined[0]}")
    print(f"  Score: {best_combined[2]:.4f} (+/- {best_combined[3]:.4f})")
    
    # Best individual feature combination
    print(f"\nBest individual feature combination:")
    print(f"  Features: {best_combo_features}")
    print(f"  Score: {best_score:.4f} (+/- {best_std:.4f})")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if best_combo_features:
        print(f"\nUse these features in your model:")
        for i, feature in enumerate(best_combo_features, 1):
            print(f"  {i}. {feature}")
        
        # Save recommended features
        with open('recommended_features.txt', 'w') as f:
            f.write('\n'.join(best_combo_features))
        print(f"\nRecommended features saved to 'recommended_features.txt'")
    

if __name__ == '__main__':
    main()
