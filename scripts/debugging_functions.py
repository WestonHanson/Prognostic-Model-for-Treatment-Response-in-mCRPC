# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: To debug data from XGBoost model

import numpy as np
from sklearn.metrics import accuracy_score

def model_debugging(model, cv_results, median_params, best_boost_round, X_train_mc, dtrain_mc_train, y_train_mc_encoded):
    """
    Parameters:
    -----------
        mc_cohort: pandas DataFrame
            A dataframe of the mc_cohort.
            
        selective_features: List
            List of features to select for.

        responder_group: String
            Name of responder group column.

    Function:
    ---------
        - Does an ANCOVA test on responder vs non-responder groups of each TFBS in selective_features.
        - Does multiple test correction for all pvalues.
        - Creates a paired box plot for responder groups and a sorted histogram for each patient. 
        
    Returns:
    --------
        void
    """
    print("\n" + "="*80)
    print("DEEP MODEL DEBUGGING")
    print("="*80)

    # 1. Check what the CV actually returned
    print("\n1. CROSS-VALIDATION RESULTS:")
    print("-" * 60)
    for i, result in enumerate(cv_results):
        print(f"\nFold {i+1}:")
        print(f"  Best params: {result['best_params']}")
        print(f"  Best round: {result['best_round']}")
        print(f"  outer auc: {result['outer_metric_scores']['auc']:.4f}")
        print(f"  Confusion matrix:\n{result['confusion_matrix']}")
        
        # Check if CV fold is also predicting all 1's
        cm = result['confusion_matrix']
        if cm.shape[0] == 2 and cm.shape[1] == 2:
            total_pred_0 = cm[0, 0] + cm[1, 0]  # All predictions of class 0
            total_pred_1 = cm[0, 1] + cm[1, 1]  # All predictions of class 1
            print(f"  Predictions: Class 0={total_pred_0}, Class 1={total_pred_1}")
            if total_pred_0 == 0:
                print("  ⚠️  WARNING: This fold also predicted all 1's!")

    # 2. Check median parameters
    print("\n2. MEDIAN PARAMETERS:")
    print("-" * 60)
    print(f"median_params: {median_params}")
    print(f"best_boost_round: {best_boost_round}")

    # Check if parameters are too restrictive
    if 'max_depth' in median_params and median_params['max_depth'] <= 2:
        print("⚠️  WARNING: max_depth is very low, model may be too simple")
    if 'min_child_weight' in median_params and median_params['min_child_weight'] >= 10:
        print("⚠️  WARNING: min_child_weight is very high, model may be too constrained")

    # 3. Test the trained model on TRAINING data
    print("\n3. MODEL PERFORMANCE ON TRAINING DATA:")
    print("-" * 60)
    train_pred_prob = model.predict(dtrain_mc_train)
    debug_median_values = np.median(train_pred_prob)
    train_pred = (train_pred_prob > debug_median_values).astype(int)

    print(f"Training set size: {len(y_train_mc_encoded)}")
    print(f"Probability stats:")
    print(f"  Min: {train_pred_prob.min():.4f}")
    print(f"  Max: {train_pred_prob.max():.4f}")
    print(f"  Mean: {train_pred_prob.mean():.4f}")
    print(f"  Std: {train_pred_prob.std():.4f}")

    train_actual = y_train_mc_encoded.ravel().astype(int)
    print(f"\nActual distribution: {np.bincount(train_actual)}")
    print(f"Predicted distribution: {np.bincount(train_pred)}")

    train_acc = accuracy_score(train_actual, train_pred)
    print(f"Training accuracy: {train_acc:.4f}")

    if train_acc < 0.6:
        print("⚠️  WARNING: Low training accuracy - model is not learning!")

    # Show some individual predictions
    print("\nFirst 20 training predictions:")
    print("Index | Actual | Prob   | Pred | Match")
    print("-" * 50)
    for i in range(min(20, len(train_actual))):
        actual = train_actual[i]
        prob = train_pred_prob[i]
        pred = train_pred[i]
        match = "✓" if actual == pred else "✗"
        print(f"{i:5d} | {actual:6d} | {prob:.4f} | {pred:4d} | {match:^5s}")

    # 4. Check if model has actually learned anything
    print("\n4. MODEL STRUCTURE CHECK:")
    print("-" * 60)

    # Get model dump
    model_dump = model.get_dump()
    num_trees = len(model_dump)
    print(f"Number of trees in model: {num_trees}")

    if num_trees == 0:
        print("⚠️  CRITICAL ERROR: Model has no trees!")
    elif num_trees < 10:
        print(f"⚠️  WARNING: Very few trees ({num_trees}), model may be undertrained")

    # Check first tree
    if num_trees > 0:
        first_tree = model_dump[0]
        print(f"\nFirst tree structure (truncated):")
        print(first_tree[:500] if len(first_tree) > 500 else first_tree)

    # 5. Get feature importance
    print("\n5. FEATURE IMPORTANCE:")
    print("-" * 60)
    try:
        feature_importance = model.get_score(importance_type='gain')
        if len(feature_importance) == 0:
            print("⚠️  WARNING: No feature importance scores!")
        else:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print(f"Top 10 features by gain:")
            for feat, score in sorted_features[:10]:
                # Map feature index back to name if possible
                if feat.startswith('f') and feat[1:].isdigit():
                    feat_idx = int(feat[1:])
                    if feat_idx < len(X_train_mc.columns):
                        feat_name = X_train_mc.columns[feat_idx]
                        print(f"  {feat_name} ({feat}): {score:.4f}")
                    else:
                        print(f"  {feat}: {score:.4f}")
                else:
                    print(f"  {feat}: {score:.4f}")
    except Exception as e:
        print(f"⚠️  Error getting feature importance: {e}")

    # 7. Check for leaf values
    print("\n7. LEAF VALUE ANALYSIS:")
    print("-" * 60)
    # Get leaf predictions for training data to see if they're all similar
    train_leaves = model.predict(dtrain_mc_train, pred_leaf=True)
    print(f"Shape of leaf predictions: {train_leaves.shape}")
    print(f"train_leaves: {train_leaves}")
    
    # Handle both 1D and 2D cases
    if train_leaves.ndim == 1:
        print("Note: Leaf predictions are 1-dimensional (single tree or specific model configuration)")
        unique_patterns = len(np.unique(train_leaves))
        print(f"Unique leaf patterns: {unique_patterns}")
        if unique_patterns <= 2:
            print("⚠️  WARNING: Very few unique leaf patterns - model is too simple!")
    else:
        print(f"Unique leaf patterns in first tree: {len(np.unique(train_leaves[:, 0]))}")
        # Check if all samples land in the same leaf
        unique_patterns = len(np.unique(train_leaves, axis=0))
        print(f"Total unique leaf patterns across all trees: {unique_patterns}")
        if unique_patterns <= 2:
            print("⚠️  WARNING: Very few unique leaf patterns - model is too simple!")

    print("\n" + "="*80)
    print("END DEEP DEBUGGING")
    print("="*80 + "\n")