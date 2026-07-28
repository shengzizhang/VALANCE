from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, make_scorer, f1_score
from sklearn.model_selection import train_test_split
import pickle
from tabpfn.finetuning import FinetunedTabPFNClassifier
import pandas as pd
import numpy as np
import h5py
import torch
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import argparse
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib

# FIX 1: removed unused imports (TabPFNClassifier, ModelVersion, interpretability,
#         RandomForestTabPFNClassifier — rf_pfn no longer exists in tabpfn-extensions)

class FinetunedTabPFNClassifierFixed(FinetunedTabPFNClassifier):
    _estimator_type = "classifier"  # class-level, survives sklearn clone()

parser = argparse.ArgumentParser()
parser.add_argument('--Ab',       type=str,   default='VRC01', help='antibody name')
parser.add_argument('--cutoff',   type=float, default=50,      help='IC50 cutoff')
parser.add_argument('--ft_epochs', type=int,  default=30,      help='fine-tuning epochs')   # FIX 2: was used but never defined
parser.add_argument('--ft_lr',    type=float, default=1e-5,    help='fine-tuning learning rate')  # FIX 2
args = parser.parse_args()

ab     = args.Ab
device = "cuda" if torch.cuda.is_available() else "cpu"  # FIX 3: torch was not imported

X = []
y = []
ordered_vids = []

with h5py.File(f"{ab}_training.h5", "r") as f:
    for sid in f.keys():
        X.append(f[sid][:])
        binary_label = 0 if float(f[sid].attrs['ic50']) < args.cutoff else 1
        y.append(binary_label)
        ordered_vids.append(sid)
print('loading training data')

X = np.array(X)
y = np.array(y)
print(f"X shape: {X.shape}  |  Class dist: {np.bincount(y)} (0=sensitive, 1=resistant)")

clf_base = FinetunedTabPFNClassifierFixed(
    device=device,
    epochs=args.ft_epochs,
    learning_rate=args.ft_lr,
)

pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('clf', clf_base)  # FIX 4: was 'rf_pfn', must match FinetunedTabPFNClassifierFixed
])

param_grid = {
    'feature_selection__k': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
}

cv_inner  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_outer  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# FIX 5: 'roc_auc' string triggers sklearn type-check failure on FinetunedTabPFNClassifierFixed;
# use a callable scorer instead (mathematically identical result)
def auc_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)

scoring_metrics = {
    'mcc': make_scorer(matthews_corrcoef),
    'acc': 'accuracy',
    'auc': auc_scorer,
    'f1':  'f1',
}

inner_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_inner,
    scoring=scoring_metrics,
    refit='mcc',
    verbose=3,
    return_train_score=False
)

results = cross_validate(
    inner_search,
    X,
    y,
    cv=cv_outer,
    scoring=scoring_metrics
)

metrics = ['mcc', 'acc', 'auc', 'f1']
formatted_row = {}
for m in metrics:
    key = f'test_{m}'
    mean_val = np.mean(results[key])
    std_val  = np.std(results[key])
    formatted_row[m.upper()] = f"{mean_val:.2f}({std_val:.2f})"

final_df = pd.DataFrame([formatted_row])
print(final_df.to_string(index=False))
final_df.to_csv(f"{ab}_{args.cutoff}_finetuned_tabpfn_nestedcv_metrics.csv", index=False)

