"""
evaluate_models.py
------------------
Comparison of centralized vs federated MLP models trained on MIMIC-IV
and evaluated on MIMIC-IV and eICU holdout sets (PoC3 — MEDomics).

Models:
    - Centralized MLP: trained on MIMIC-IV via MEDomics (PyCaret pipeline)
    - Federated MLP:   trained on 9 eICU hospitals via MEDfl

Outputs (saved to results/):
    - resultats_comparaison.csv     : full metrics table
    - figures/cm_*.png              : confusion matrices
    - figures/shap_*.png            : SHAP feature importance (federated model)

Usage:
    pip install shap matplotlib scikit-learn pandas numpy
    python evaluate_models.py

Requirements:
    Place the following files in the working directory:
    - model_centralise.pkl
    - model_fed.pkl
    - Holdout_MIMIC.csv
    - Holdout_eicu_afterdividing 1.csv
    - Holdout_eicu_dataset_hospital_*.csv  (9 hospitals)

Note:
    MIMIC-IV and eICU data require credentialed PhysioNet access.
    See: https://physionet.org
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, recall_score,
    precision_score, f1_score, accuracy_score,
    confusion_matrix
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not installed. Run: pip install shap")
    SHAP_AVAILABLE = False

# Compatibility patch for pandas >= 2.0
if not hasattr(pd.Series, "iteritems"):
    pd.Series.iteritems = pd.Series.items
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items


# ================================================================
# CONFIGURATION
# ================================================================
TARGET               = 'deceased'
DATA_DIR             = '.'
MODEL_FED            = 'model_fed.pkl'
MODEL_CEN            = 'model_centralise.pkl'
THRESHOLD_CENTRALISE = 0.2
OUTPUT_DIR           = 'results'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

# Evaluation datasets: MIMIC holdout + 9 eICU hospitals
DATASETS = {
    'MIMIC holdout'  : 'Holdout_MIMIC.csv',
    'eICU aggregated': 'Holdout_eicu_afterdividing 1.csv',
    'Hospital 122'   : 'Holdout_eicu_dataset_hospital_122.csv',
    'Hospital 167'   : 'Holdout_eicu_dataset_hospital_167.csv',
    'Hospital 188'   : 'Holdout_eicu_dataset_hospital_188.csv',
    'Hospital 199'   : 'Holdout_eicu_dataset_hospital_199.csv',
    'Hospital 252'   : 'Holdout_eicu_dataset_hospital_252.csv',
    'Hospital 264'   : 'Holdout_eicu_dataset_hospital_264.csv',
    'Hospital 338'   : 'Holdout_eicu_dataset_hospital_338.csv',
    'Hospital 420'   : 'Holdout_eicu_dataset_hospital_420.csv',
    'Hospital 73'    : 'Holdout_eicu_dataset_hospital_73.csv',
}

# Hospitals selected for SHAP analysis
SHAP_DATASETS = {
    'Hospital 264': 'Holdout_eicu_dataset_hospital_264.csv',
    'Hospital 420': 'Holdout_eicu_dataset_hospital_420.csv',
    'Hospital 188': 'Holdout_eicu_dataset_hospital_188.csv',
    'Hospital 167': 'Holdout_eicu_dataset_hospital_167.csv',
}


# ================================================================
# MODEL LOADING
# ================================================================
print("Loading models...")
with open(MODEL_FED, 'rb') as f:
    model_federe = pickle.load(f)
with open(MODEL_CEN, 'rb') as f:
    model_centralise = pickle.load(f)
print("  Models loaded successfully.")


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def inject_federated_weights(model):
    """
    Inject federated weights (stored in actual_estimator.coefs_)
    into classifier_ for inference.

    This is required because MEDfl stores aggregated weights separately
    from the base PyCaret classifier object.
    """
    actual_estimator = model.steps[-1][1]
    actual_estimator.classifier_.coefs_      = actual_estimator.coefs_
    actual_estimator.classifier_.intercepts_ = actual_estimator.intercepts_
    return actual_estimator.classifier_


def get_centralised_mlp(model):
    """Return the trained centralized MLP classifier."""
    return model.steps[-1][1].classifier_


def prepare_data(df, model, target=TARGET):
    """
    Apply PyCaret pipeline preprocessing steps to a DataFrame.
    All steps except the final estimator are applied.

    Parameters
    ----------
    df     : Input DataFrame (holdout set).
    model  : Fitted PyCaret pipeline.
    target : Name of the target column.

    Returns
    -------
    X : Preprocessed feature matrix.
    y : Target series.
    """
    temp_df = df.copy()
    temp_df.columns = temp_df.columns.str.strip().str.replace(" ", "_")

    for col in ["_id", "id"]:
        if col in temp_df.columns:
            temp_df.drop(columns=[col], inplace=True, errors="ignore")

    for name, transformer in model.steps[:-1]:
        if hasattr(transformer, 'transform'):
            try:
                temp_df = transformer.transform(temp_df)
            except Exception as e:
                print(f"  Warning [{name}]: {e}")

    temp_df.dropna(how="any", inplace=True)

    if target not in temp_df.columns:
        raise ValueError(f"Target column '{target}' not found after preprocessing.")

    X = temp_df.drop(columns=[target])
    y = temp_df[target]

    # Align columns with those expected by the MLP
    mlp = model.steps[-1][1].classifier_
    if hasattr(mlp, "feature_names_in_"):
        X = X.reindex(columns=list(mlp.feature_names_in_), fill_value=0)

    return X, y


def evaluate(y_true, proba, threshold, label, verbose=True):
    """
    Compute classification metrics for a given threshold.

    Returns a dict with AUC, Accuracy, Recall, Precision, F1,
    and confusion matrix counts.
    """
    pred = (proba >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    except ValueError:
        return None

    total = len(y_true)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float('nan')

    metrics = {
        'Dataset'   : label.split(' -- ')[0],
        'Model'     : label.split(' -- ')[1] if ' -- ' in label else label,
        'Threshold' : round(threshold, 3),
        'N'         : total,
        'AUC'       : round(auc, 3),
        'Accuracy'  : round(accuracy_score(y_true, pred), 3),
        'Recall'    : round(recall_score(y_true, pred, zero_division=0), 3),
        'Precision' : round(precision_score(y_true, pred, zero_division=0), 3),
        'F1'        : round(f1_score(y_true, pred, zero_division=0), 3),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

    if verbose:
        print(f"\n  --- {label} [threshold={threshold:.3f}, n={total}] ---")
        print(f"  AUC={metrics['AUC']}  Acc={metrics['Accuracy']}  "
              f"Recall={metrics['Recall']}  Prec={metrics['Precision']}  "
              f"F1={metrics['F1']}")
        print(f"  Confusion: TN={tn}({tn/total*100:.1f}%)  "
              f"FP={fp}({fp/total*100:.1f}%)  "
              f"FN={fn}({fn/total*100:.1f}%)  "
              f"TP={tp}({tp/total*100:.1f}%)")

    return metrics


def plot_confusion_matrix(y_true, proba, threshold, title, filepath):
    """Save a confusion matrix figure."""
    pred = (proba >= threshold).astype(int)
    cm   = confusion_matrix(y_true, pred)
    total = len(y_true)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes    = ['Negative (0)', 'Positive (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    ax.set_ylabel('True label', fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_shap_importance(mlp, X, dataset_name, filepath):
    """
    Compute and save SHAP feature importance for the federated MLP.
    Uses KernelExplainer (model-agnostic) with a background sample.
    """
    if not SHAP_AVAILABLE:
        print("  SHAP not available -- skipping.")
        return

    print(f"  Computing SHAP for {dataset_name}...")
    try:
        background   = shap.sample(X, min(50, len(X)), random_state=42)
        explainer    = shap.KernelExplainer(
            lambda x: mlp.predict_proba(x)[:, 1], background
        )
        shap_values  = explainer.shap_values(X, nsamples=100)
        mean_shap    = np.abs(shap_values).mean(axis=0)
        feature_names = list(X.columns)

        sorted_idx = np.argsort(mean_shap)[::-1]
        top_n      = min(10, len(feature_names))
        top_idx    = sorted_idx[:top_n]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(range(top_n), mean_shap[top_idx][::-1], color='steelblue')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=11)
        ax.set_xlabel('Mean |SHAP value|', fontsize=11)
        ax.set_title(f'Feature Importance (SHAP) -- {dataset_name}\n(Federated model)',
                     fontsize=12)

        for bar, idx in zip(bars, top_idx[::-1]):
            ax.text(bar.get_width() + mean_shap.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{mean_shap[idx]:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

        print(f"  Top 5 features:")
        for rank, idx in enumerate(top_idx[:5], 1):
            print(f"    {rank}. {feature_names[idx]} : {mean_shap[idx]:.4f}")

    except Exception as e:
        print(f"  SHAP error: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# INJECT FEDERATED WEIGHTS
# ================================================================
print("\nInjecting federated weights...")
mlp_federe     = inject_federated_weights(model_federe)
mlp_centralise = get_centralised_mlp(model_centralise)

diff = np.abs(mlp_federe.coefs_[0] - mlp_centralise.coefs_[0])
if diff.mean() > 1e-6:
    print(f"  Federated weights confirmed (mean diff: {diff.mean():.6f}).")
else:
    print("  WARNING: weights are identical -- check federated model pkl.")


# ================================================================
# EVALUATION LOOP
# ================================================================
all_results = []

for dataset_name, filename in DATASETS.items():
    print(f"\n{'#'*60}")
    print(f"# {dataset_name}")
    print(f"{'#'*60}")

    try:
        df = pd.read_csv(f"{DATA_DIR}/{filename}")
        print(f"  Loaded: {len(df)} rows")

        X, y = prepare_data(df, model_federe, TARGET)
        print(f"  After preprocessing: {len(y)} patients | "
              f"positive rate: {y.mean()*100:.1f}%")

        proba_f = mlp_federe.predict_proba(X)[:, 1]
        proba_c = mlp_centralise.predict_proba(X)[:, 1]

        diff = np.abs(proba_f - proba_c)
        print(f"  Proba diff: mean={diff.mean():.4f}  max={diff.max():.4f}")

        # Youden threshold for federated model
        try:
            fpr, tpr, thresholds = roc_curve(y, proba_f)
            opt_threshold = float(thresholds[np.argmax(tpr - fpr)])
        except Exception:
            opt_threshold = 0.5
        print(f"  Youden threshold (federated): {opt_threshold:.3f}")

        # Metrics for 3 configurations
        r1 = evaluate(y, proba_c, THRESHOLD_CENTRALISE,
                      f"{dataset_name} -- Centralised (threshold={THRESHOLD_CENTRALISE})")
        r2 = evaluate(y, proba_f, THRESHOLD_CENTRALISE,
                      f"{dataset_name} -- Federated (threshold={THRESHOLD_CENTRALISE})")
        r3 = evaluate(y, proba_f, opt_threshold,
                      f"{dataset_name} -- Federated (Youden={opt_threshold:.3f})")

        for r in [r1, r2, r3]:
            if r is not None:
                all_results.append(r)

        # Confusion matrices
        safe_name = dataset_name.replace(' ', '_')

        plot_confusion_matrix(
            y, proba_c, THRESHOLD_CENTRALISE,
            f"Confusion Matrix -- {dataset_name}\nCentralised (threshold=0.2)",
            f"{OUTPUT_DIR}/figures/cm_{safe_name}_centralised.png"
        )
        plot_confusion_matrix(
            y, proba_f, opt_threshold,
            f"Confusion Matrix -- {dataset_name}\n"
            f"Federated (Youden threshold={opt_threshold:.3f})",
            f"{OUTPUT_DIR}/figures/cm_{safe_name}_federated.png"
        )

    except FileNotFoundError:
        print(f"  File not found: {filename}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


# ================================================================
# SHAP FEATURE IMPORTANCE
# ================================================================
print(f"\n{'='*60}")
print(" SHAP FEATURE IMPORTANCE -- 4 SELECTED HOSPITALS")
print(f"{'='*60}")

for dataset_name, filename in SHAP_DATASETS.items():
    print(f"\n--- SHAP: {dataset_name} ---")
    try:
        df   = pd.read_csv(f"{DATA_DIR}/{filename}")
        X, y = prepare_data(df, model_federe, TARGET)
        safe_name = dataset_name.replace(' ', '_')
        plot_shap_importance(
            mlp_federe, X, dataset_name,
            f"{OUTPUT_DIR}/figures/shap_{safe_name}_federated.png"
        )
    except FileNotFoundError:
        print(f"  File not found: {filename}")
    except Exception as e:
        print(f"  SHAP error: {e}")


# ================================================================
# SUMMARY TABLE
# ================================================================
print(f"\n\n{'='*60}")
print(" FINAL RESULTS SUMMARY")
print(f"{'='*60}")

if all_results:
    df_results = pd.DataFrame(all_results)
    cols = ['Dataset', 'Model', 'Threshold', 'N',
            'AUC', 'Accuracy', 'Recall', 'Precision', 'F1',
            'TN', 'FP', 'FN', 'TP']
    df_results = df_results[cols]
    print(df_results.to_string(index=False))

    output_csv = f'{OUTPUT_DIR}/resultats_comparaison.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")

    print(f"\n{'='*80}")
    print(" SUMMARY: Centralised vs Federated (Youden threshold)")
    print(f"{'='*80}")
    print(f"\n{'Dataset':<20} {'AUC Cen':>8} {'AUC Fed':>8} {'dAUC':>7} "
          f"{'Rec Cen':>9} {'Rec Fed':>9} {'dRec':>7} {'F1 Cen':>8} {'F1 Fed':>8}")
    print("-" * 85)

    for ds in DATASETS.keys():
        rows_cen = df_results[
            (df_results['Dataset'] == ds) &
            (df_results['Model'].str.contains('Centralised'))
        ]
        rows_fed = df_results[
            (df_results['Dataset'] == ds) &
            (df_results['Model'].str.contains('Youden'))
        ]
        if not rows_cen.empty and not rows_fed.empty:
            auc_c = rows_cen['AUC'].values[0]
            auc_f = rows_fed['AUC'].values[0]
            rec_c = rows_cen['Recall'].values[0]
            rec_f = rows_fed['Recall'].values[0]
            f1_c  = rows_cen['F1'].values[0]
            f1_f  = rows_fed['F1'].values[0]
            d_auc = auc_f - auc_c
            d_rec = rec_f - rec_c
            s_auc = "+" if d_auc > 0.001 else ("-" if d_auc < -0.001 else "=")
            s_rec = "+" if d_rec > 0.001 else ("-" if d_rec < -0.001 else "=")
            print(f"{ds:<20} {auc_c:>8} {auc_f:>8} {s_auc}{abs(d_auc):>6.3f} "
                  f"{rec_c:>9} {rec_f:>9} {s_rec}{abs(d_rec):>6.3f} "
                  f"{f1_c:>8} {f1_f:>8}")

print(f"\nFigures saved to '{OUTPUT_DIR}/figures/'")
print("  cm_*_centralised.png  : centralised model confusion matrices")
print("  cm_*_federated.png    : federated model confusion matrices")
print("  shap_*_federated.png  : federated model SHAP feature importance")
