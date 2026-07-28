from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split
import pickle
from tabpfn.finetuning import FinetunedTabPFNClassifier
import pandas as pd
import numpy as np
import h5py
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import argparse
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib
import torch


class FinetunedTabPFNClassifierFixed(FinetunedTabPFNClassifier):
    _estimator_type = "classifier"  # class-level, survives sklearn clone()

parser = argparse.ArgumentParser()
parser.add_argument('--Ab',              type=str,   default='VRC01', help='antibody name')
parser.add_argument('--cutoff',          type=float, default=50,      help='IC50 cutoff')
parser.add_argument('--traindata',       type=str,   default='env_neu_unique_ab_removed_outliers_duplicates_geomean_include_TBDs.txt')
parser.add_argument('--train_embedding', type=str,   default='',      help='embedding file from generate_embedding.py')
parser.add_argument('--tabpfn_model',    type=str,   default='./tabpfn-v2.5-classifier-v2.5_default.ckpt')
parser.add_argument('--ft_epochs',       type=int,   default=30,      help='fine-tuning epochs')
parser.add_argument('--ft_lr',           type=float, default=1e-5,    help='fine-tuning learning rate')
args = parser.parse_args()

ab     = args.Ab
cutoff = args.cutoff
device = "cuda" if torch.cuda.is_available() else "cpu"

X = []
y = []

df = pd.read_csv(args.traindata, sep='\t')
df.dropna(subset=['IC50'], inplace=True)
df.loc[:, 'IC50'] = df['IC50'].apply(pd.to_numeric)
Ab_df = df.loc[df['Antibody'].str.match('^' + ab + '$', na=True)]
print(f"Total {ab} training dataset: {len(Ab_df)}")
ic50_lookup = Ab_df.set_index('Virus')['IC50'].to_dict()

with h5py.File(args.train_embedding, "r") as f:
    for sid in f.keys():
        X.append(f[sid][:])
        raw_ic50 = ic50_lookup.get(sid, None)
        binary_label = 0 if float(raw_ic50) < cutoff else 1
        y.append(binary_label)

X = np.array(X)
y = np.array(y)
print("data loaded")


clf_base = FinetunedTabPFNClassifierFixed(
    device=device,
    epochs=args.ft_epochs,
    learning_rate=args.ft_lr,
)


pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('clf', clf_base),
])
param_grid = {
    'feature_selection__k': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def auc_scorer(estimator, X, y):
    proba = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba)

scoring_metrics = {
    'mcc': make_scorer(matthews_corrcoef),
    'acc': 'accuracy',
    'auc': auc_scorer,   # callable scorer, no type checking, calls predict_proba directly
}

clf = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring=scoring_metrics,
    refit='mcc',
    verbose=3,
    return_train_score=False
)

clf.fit(X, y)

results_df = pd.DataFrame(clf.cv_results_)
summary = results_df[[
    'param_feature_selection__k',
    'mean_test_mcc', 'std_test_mcc',
    'mean_test_acc', 'std_test_acc',
    'mean_test_auc', 'std_test_auc'
]].copy()
summary.columns = ['k', 'mcc_avg', 'mcc_sd', 'acc_avg', 'acc_sd', 'auc_avg', 'auc_sd']
summary = summary.sort_values(by='mcc_avg', ascending=False)

best_k = clf.best_params_['feature_selection__k']
print(f"Best k found: {best_k} with Mean MCC: {clf.best_score_:.4f}")

summary.to_csv(f'{ab}_{cutoff}_cv_results_finetuned_tabpfn.csv', index=False)
final_model = clf.best_estimator_
joblib.dump(final_model, f"{ab}_{cutoff}_final_model_finetuned_tabpfn.joblib", compress=3)