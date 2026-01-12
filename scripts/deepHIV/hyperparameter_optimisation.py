import sys
sys.path.append('/home/yujieq/work/ML_training/deep_hiv_ab_pred')
import numpy as np
import optuna
import torch as t
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import HYPERPARAM_FOLDER_ANTIBODIES
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json, get_experiment
import os
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    CV_FOLDS_TRIM, N_TRIALS, PRUNE_TREHOLD, ANTIBODIES_LIST, FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL_BUT_LAST_LAYER, FREEZE_EMBEDDINGS_ONLY
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF, FINAL_MODEL
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import parse_catnap_sequences_to_embeddings
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net, pseudo_pretrain_net, load_pretrain_net, cross_validate_antibody, cross_validate_antibody_predict, cross_validate_antibody_focal_loss, cross_validate_antibody_with_oversampler
from os.path import join
import mlflow
import statistics
from deep_hiv_ab_pred.util.metrics import log_metrics_per_cv_antibody
import copy
from deep_hiv_ab_pred.util.logging import setup_logging
import logging
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState
import matplotlib.pyplot as plt
from deep_hiv_ab_pred.model.FC_GRU_ATT import get_FC_GRU_ATT_model
from deep_hiv_ab_pred.training.training import train_with_frozen_antibody_and_embedding_GRU, train_with_frozen_antibody_and_embedding, train_with_frozen_embeddings_only
from imblearn.over_sampling import RandomOverSampler

# Set the MLflow tracking URI  
mlflow.set_tracking_uri('/home/yujieq/work/ML_training/deep_hiv_ab_pred/mlruns') 


def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label=f'Training Loss CV {i + 1}')
        plt.plot(epochs, val_loss, label=f'Validation Loss CV {i + 1}', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_mcc(train_mcc_CVs, val_mcc_CVs, save_path):
    plt.figure(figsize=(10, 5))
    for i, (train_mcc, val_mcc) in enumerate(zip(train_mcc_CVs, val_mcc_CVs)):
        epochs = range(1, len(train_mcc) + 1)
        plt.plot(epochs, train_mcc, label=f'Training MCC CV {i + 1}')
        plt.plot(epochs, val_mcc, label=f'Validation MCC CV {i + 1}', linestyle='--')
    plt.title('Training and Validation MCC')
    plt.xlabel('Epochs')
    plt.ylabel('MCC')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def propose_conf_for_frozen_antb_and_embeddings(trial: optuna.trial.Trial, base_conf: dict):
    return {
        'BATCH_SIZE': trial.suggest_int('BATCH_SIZE', 50, 5000),
        'LEARNING_RATE': trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-1),
        'GRAD_NORM_CLIP': trial.suggest_loguniform('GRAD_NORM_CLIP', 1e-2, 1000),
        'FULLY_CONNECTED_DROPOUT': trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5),
        'EMBEDDING_DROPOUT': base_conf['EMBEDDING_DROPOUT'],
        'KMER_LEN_VIRUS': base_conf['KMER_LEN_VIRUS'],
        'KMER_STRIDE_VIRUS': base_conf['KMER_STRIDE_VIRUS'],
        'RNN_HIDDEN_SIZE': base_conf['RNN_HIDDEN_SIZE'],
        'ANTIBODIES_DROPOUT': base_conf['ANTIBODIES_DROPOUT'],
        'EPOCHS': 100
    }

class CrossValidationPruner(BasePruner):
    def __init__(self, treshold):
        self.treshold = treshold

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        step = trial.last_step
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if not completed_trials:
            return False
        score_matrix = np.array([
            [ t.intermediate_values[i] for i in range(len(t.intermediate_values)) ]
            for t in completed_trials
        ])
        global_average = np.zeros(len(score_matrix))
        trail_average = 0
        for i in range(step + 1):
            global_average = global_average + score_matrix[:, i]
            trail_average = trail_average + trial.intermediate_values[i]
        global_average = global_average / (step + 1)
        trail_average = trail_average / (step + 1)
        maximum = max(global_average)
        return trail_average < maximum - self.treshold

def propose_conf_for_frozen_net_without_last_layer(trial: optuna.trial.Trial, base_conf: dict):
    conf = copy.deepcopy(base_conf)
    conf['GRAD_NORM_CLIP'] = 1000
    conf['BATCH_SIZE'] = trial.suggest_int('BATCH_SIZE', 50, 5000)
    conf['EPOCHS'] = trial.suggest_int('EPOCHS', 1, 100)
    conf['LEARNING_RATE'] = trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-1)
    conf['FULLY_CONNECTED_DROPOUT'] = trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5)
    return conf

# def get_objective_cross_validation(antibody, cv_folds_trim, freeze_mode, pretrain_epochs, splits_file):
#     all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)
#     splits = all_splits[antibody]
#     if not os.path.isfile(os.path.join(MODELS_FOLDER, f'model_our_data_{antibody}_pretrain.tar')):
#         pretrain_net(antibody, splits['pretraining'], catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, pretrain_epochs)
#     def objective(trial):
#         if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
#             conf = propose_conf_for_frozen_antb_and_embeddings(trial, base_conf)
#         # deprecated
#         elif freeze_mode == FREEZE_ALL_BUT_LAST_LAYER:
#             conf = propose_conf_for_frozen_net_without_last_layer(trial, base_conf)
#         else:
#             raise 'Must provide a proper freeze mode.'
#         try:
#             cv_metrics = cross_validate_antibody(antibody, splits['cross_validation'], catnap, conf, virus_seq,
#                 virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, trial, cv_folds_trim, 0, freeze_mode)
#             cv_metrics = np.array(cv_metrics)
#             cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
#             return cv_mean_mcc
#         except optuna.TrialPruned as pruneError:
#             raise pruneError
#         except Exception as e:
#             if str(e).startswith('CUDA out of memory'):
#                 logging.error('CUDA out of memory', exc_info = True)
#                 # t.cuda.empty_cache()
#                 raise optuna.TrialPruned()
#             elif 'CUDA error' in str(e):


#                 logging.error('CUDA error', exc_info = True)
#                 # t.cuda.empty_cache()
#                 raise optuna.TrialPruned()
#             logging.exception(str(e), exc_info = True)
#             logging.error(f'Configuration {conf}')
#             raise optuna.TrialPruned()
#         return cv_mean_mcc
#     return objective


def get_objective_cross_validation(antibody, cv_folds_trim, freeze_mode, pretrain_epochs, splits_file):
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)
    splits = all_splits[antibody]
    pretrain_net(antibody, splits['pretraining'], catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, pretrain_epochs)
    # load_pretrain_net(antibody, splits['pretraining'], catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, pretrain_epochs)
    def objective(trial):
        # if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
        #     conf = propose_conf_for_frozen_antb_and_embeddings(trial, base_conf)
        if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS or freeze_mode == FREEZE_EMBEDDINGS_ONLY:
            conf = propose_conf_for_frozen_antb_and_embeddings(trial, base_conf)
        # deprecated
        elif freeze_mode == FREEZE_ALL_BUT_LAST_LAYER:
            conf = propose_conf_for_frozen_net_without_last_layer(trial, base_conf)
        else:
            raise 'Must provide a proper freeze mode.'
        try:
            cv_metrics, train_mcc_CVs, val_mcc_CVs, train_losses, val_losses = cross_validate_antibody(antibody, splits['cross_validation'], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, trial, cv_folds_trim, 0, freeze_mode)
            # plot_loss(train_losses, val_losses, f'/home/yujieq/work/ML_training/deep_hiv_ab_pred/{antibody}_trial_{trial.number}_loss.png')
            # plot_mcc(train_mcc_CVs, val_mcc_CVs, f'/home/yujieq/work/ML_training/deep_hiv_ab_pred/{antibody}_trial_{trial.number}_mcc.png')
            cv_metrics = np.array(cv_metrics)
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
            return cv_mean_mcc
        except optuna.TrialPruned as pruneError:
            raise pruneError
        except Exception as e:
            if str(e).startswith('CUDA out of memory'):
                logging.error('CUDA out of memory', exc_info = True) 
                # t.cuda.empty_cache()
                raise optuna.TrialPruned()
            elif 'CUDA error' in str(e):
                logging.error('CUDA error', exc_info = True)
                # t.cuda.empty_cache()
                raise optuna.TrialPruned()
            logging.exception(str(e), exc_info = True)
            logging.error(f'Configuration {conf}')
            raise optuna.TrialPruned()
        return cv_mean_mcc
    return objective

def optimize_hyperparameters(antibody_name, splits_file, cv_folds_trim = 10, n_trials = 100, prune_trehold = .05, model_trial_name = '',
        freeze_mode = FREEZE_EMBEDDINGS_ONLY, pretrain_epochs=None):
    pruner = CrossValidationPruner(prune_trehold)
    study_name = f'Finetune_1000_threshold0.2_removed_outliers_duplicates_geomean_10E8_lineage_forCV_july16_data_{model_trial_name}_{antibody_name}'
    study = optuna.create_study(study_name = study_name, direction = 'maximize',
                                storage = f'sqlite:///{study_name}.db', load_if_exists = True, pruner = pruner)
    objective = get_objective_cross_validation(antibody_name, cv_folds_trim, freeze_mode, pretrain_epochs, splits_file)
    study.optimize(objective, n_trials = n_trials)
    logging.info(study.best_params)
    dump_json(study.best_params, join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold0.2_removed_outliers_duplicates_geomean_10E8_lineage_forCV_{antibody_name}.json'))

def get_data(splits_file):
    all_splits = read_json_file(splits_file)
    catnap = read_json_file(CATNAP_FLAT)
    base_conf = read_json_file(DEFAULT_CONF)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
        base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS']
    )
    return all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq

def add_properties_from_base_config(conf, base_conf):
    for prop in base_conf:
        if prop not in conf:
            conf[prop] = base_conf[prop]
    return conf


def test_optimized_antibody(antibody, splits_file, model_trial_name = '', freeze_mode = FREEZE_EMBEDDINGS_ONLY, pretrain_epochs = None):
    mlflow.log_params({ 'cv_folds_trim': CV_FOLDS_TRIM, 'n_trials': N_TRIALS, 'prune_trehold': PRUNE_TREHOLD })
    optimize_hyperparameters(antibody, splits_file, cv_folds_trim = CV_FOLDS_TRIM, n_trials = N_TRIALS, prune_trehold = PRUNE_TREHOLD,
        model_trial_name = model_trial_name, freeze_mode = freeze_mode, pretrain_epochs = pretrain_epochs)
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)   
    mlflow.log_artifact(DEFAULT_CONF, 'base_conf.json')
    conf = read_json_file(join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold0.2_removed_outliers_duplicates_geomean_10E8_lineage_forCV_{antibody}.json'))
    mlflow.log_artifact(join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold0.2_removed_outliers_duplicates_geomean_10E8_lineage_forCV_{antibody}.json'), f'{antibody} conf.json')
    conf = add_properties_from_base_config(conf, base_conf)
    print(conf)
    cv_metrics, train_mcc_CVs, val_mcc_CVs, train_losses, val_losses = cross_validate_antibody(antibody, all_splits[antibody]['cross_validation'], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, cv_folds_skip = 0, cv_folds_trim = 100, freeze_mode = freeze_mode)
    plot_loss(train_losses, val_losses, f'/home/yujieq/work/ML_training/deep_hiv_ab_pred/{antibody}_CV_threshold0.2_removed_outliers_duplicates_geomean_10E8_lineage_forCV_loss.png')
    plot_mcc(train_mcc_CVs, val_mcc_CVs, f'/home/yujieq/work/ML_training/deep_hiv_ab_pred/{antibody}_CV_threshold0.2_removed_outliers_duplicates_geomean_10E8_lineage_forCV_mcc.png')
    cv_mean_acc, cv_mean_mcc, cv_mean_auc = log_metrics_per_cv_antibody(cv_metrics, antibody)
    # cv_mean_acc, cv_mean_mcc, cv_mean_auc = 0, 0, 0
    return cv_mean_acc, cv_mean_mcc, cv_mean_auc



def test_optimized_antibody_final_model(antibody, splits_file, model_trial_name = '', freeze_mode = FREEZE_EMBEDDINGS_ONLY, pretrain_epochs = None):
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)   
    mlflow.log_artifact(DEFAULT_CONF, 'base_conf.json')
    conf = read_json_file(join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_VRC07-523-W54-LS.v3.json'))
    mlflow.log_artifact(join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_VRC07-523-W54-LS.v3.json'), f'{antibody} conf.json')
    conf = add_properties_from_base_config(conf, base_conf)
    print(conf)
    # train_final_model_best1_of100(antibody, splits_file, conf)
    # train_final_model_all10folds_of_best_repeat(antibody, splits_file, conf)
    train_final_model_5folds_1_repeat(antibody, splits_file, conf)
    cv_mean_acc, cv_mean_mcc, cv_mean_auc = 0, 0, 0
    return cv_mean_acc, cv_mean_mcc, cv_mean_auc


# def train_final_model(antibody,splits_file,conf):    
#     # conf = read_json_file(join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody}.json'))
#     # conf = add_properties_from_base_config(conf, base_conf)
#     print(conf)
#     all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)
#     training = [a for a in catnap if a[1] == antibody]
#     # rest_assays = [a for a in catnap if a[1] != antibody]
#     #assert len(pretraining_assays) == len(splits_pretraining)
#     train_set = AssayDataset(training, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#     # val_set = AssayDataset(rest_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#     loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
#     # loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
#     model = get_FC_GRU_ATT_model(conf)
#     checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold0.2_july16_data_{antibody}_pretrain.tar'))
#     model.load_state_dict(checkpoint['model'])
#     _, _, metrics = train_with_frozen_antibody_and_embedding_GRU(
#         model, conf, loader_train, loader_train, None, 100, f'model_threshold0.2_july16_data_{antibody}', FINAL_MODEL, True, log_every_epoch = True
#     )   
    
#     print(f"{antibody} training done\nMCC:{metrics[0]}\nAUC:{metrics[1]}\nACC:{metrics[2]}\n")
#     return metrics



def train_final_model_best1_of100(antibody, splits_file, conf):
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)
    cross_validation_folds = all_splits[antibody]['cross_validation']
    
    all_metrics = []  # To store metrics from each fold
    best_metrics = None
    best_fold_idx = -1
    best_model_state = None
    
    for fold_idx, fold in enumerate(cross_validation_folds):
        print(f"\nProcessing fold {fold_idx + 1}/{len(cross_validation_folds)}")
        
        # Get train and test IDs for this specific fold
        train_ids = set(fold['train'])
        test_ids = set(fold['test'])
        
        # Filter catnap for this fold's train/test sets
        training = [a for a in catnap if a[0] in train_ids]
        testing = [a for a in catnap if a[0] in test_ids]
        
        print(f"Number of training instances in this fold: {len(training)}")
        print(f"Number of testing instances in this fold: {len(testing)}")

        # Check for single-class test set
        test_labels = [a[3] for a in testing]  # Assuming ground truth is at index 3
        train_labels = [a[3] for a in training]
        
        print(f"Fold {fold_idx + 1}: Train samples - {len(training)} (Pos: {sum(train_labels)}, Neg: {len(train_labels)-sum(train_labels)})")
        print(f"Fold {fold_idx + 1}: Test samples - {len(testing)} (Pos: {sum(test_labels)}, Neg: {len(test_labels)-sum(test_labels)})")
        
        # Skip fold if test set has only one class
        if len(set(test_labels)) < 2:
            print(f"Fold {fold_idx + 1}: Skipping due to single-class test labels")
            continue

        
        # Create datasets and loaders for this fold
        train_set = AssayDataset(training, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        val_set = AssayDataset(testing, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        
        # Initialize model for this fold
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold0.2_epitope_with_all_lineage_july16_data_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        
        # Train and evaluate on this fold (don't save during training)
        _, _, metrics = train_with_frozen_antibody_and_embedding_GRU(
            model, conf, loader_train, loader_val, fold_idx, 100, 
            f'model_threshold0.2_{antibody}_fold{fold_idx}', FINAL_MODEL, 
            save_model=False,  # Disable saving during training
            log_every_epoch=True
        )
        
        print(f"Fold {fold_idx + 1} results for {antibody}:")
        print(f"MCC: {metrics[1]}\nAUC: {metrics[2]}\nACC: {metrics[0]}\n")
        all_metrics.append(metrics)
        
        # Track best model across folds
        if best_metrics is None or metrics[MATTHEWS_CORRELATION_COEFFICIENT] > best_metrics[MATTHEWS_CORRELATION_COEFFICIENT]:
            best_metrics = metrics
            best_fold_idx = fold_idx
            best_model_state = model.state_dict()
    
    # Save only the best model after all folds are complete
    if best_model_state is not None:
        t.save({'model': best_model_state}, 
              os.path.join(FINAL_MODEL, f'model_threshold0.2_epitope_with_all_lineage_july16_data_{antibody}_best_fold{best_fold_idx}.tar'))
        print(f"Saved best model from fold {best_fold_idx + 1} with MCC: {best_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}")
    
    # Calculate and return average metrics across all folds
    avg_metrics = [sum(m[i] for m in all_metrics)/len(all_metrics) for i in range(len(metrics))]
    print(f"\nAverage across all folds for {antibody}:")
    print(f"MCC: {avg_metrics[1]}\nAUC: {avg_metrics[2]}\nACC: {avg_metrics[0]}\n")
    
    return avg_metrics, best_metrics

def train_final_model_all10folds_of_best_repeat(antibody, splits_file, conf):
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)
    cross_validation_folds = all_splits[antibody]['cross_validation']
    
    all_metrics = []  # To store metrics from each fold
    repeat_metrics = {}  # Store average metrics for each repeat
    repeat_models = {}  # Store all models for each repeat
    
    # Group folds by repeat (assuming 10 folds per repeat, 10 repeats total)
    folds_per_repeat = len(cross_validation_folds) // 10
    
    for fold_idx, fold in enumerate(cross_validation_folds):
        repeat_idx = fold_idx // folds_per_repeat  # Determine which repeat this fold belongs to
        
        print(f"\nProcessing fold {fold_idx + 1}/{len(cross_validation_folds)} (Repeat {repeat_idx + 1})")
        
        # Get train and test IDs for this specific fold
        train_ids = set(fold['train'])
        test_ids = set(fold['test'])
        
        # Filter catnap for this fold's train/test sets
        training = [a for a in catnap if a[0] in train_ids]
        testing = [a for a in catnap if a[0] in test_ids]
        
        print(f"Number of training instances in this fold: {len(training)}")
        print(f"Number of testing instances in this fold: {len(testing)}")

        # Check for single-class test set
        test_labels = [a[3] for a in testing]
        train_labels = [a[3] for a in training]
        
        print(f"Fold {fold_idx + 1}: Train samples - {len(training)} (Pos: {sum(train_labels)}, Neg: {len(train_labels)-sum(train_labels)})")
        print(f"Fold {fold_idx + 1}: Test samples - {len(testing)} (Pos: {sum(test_labels)}, Neg: {len(test_labels)-sum(test_labels)})")
        
        # Skip fold if test set has only one class
        if len(set(test_labels)) < 2:
            print(f"Fold {fold_idx + 1}: Skipping due to single-class test labels")
            continue

# ## Use below ONLY for ROS models
# ########################################################################################
# #######################################################################################
#         # Prepare features and labels for oversampling
#         train_features = [a[:3] for a in training]  # Extract features (antibody, virus, etc.)
#         train_labels = [int(a[3]) for a in training]  # Ensure labels are integers (0/1)

#         # Apply RandomOverSampler
#         ros = RandomOverSampler(random_state=42)
#         oversampled_features, oversampled_labels = ros.fit_resample(train_features, train_labels)

#         # Rebuild the oversampled dataset
#         oversampled_assays = [
#             (*features, label) for features, label in zip(oversampled_features, oversampled_labels)
#         ]

#         print(f"Fold {fold_idx + 1}: Train positive samples: {sum(train_labels)}, Train negative samples: {len(train_labels) - sum(train_labels)}")
#         print(f"Fold {fold_idx + 1}: Oversampled Train positive samples: {sum(oversampled_labels)}, Train negative samples: {len(oversampled_labels) - sum(oversampled_labels)}")
#         print(f"Fold {fold_idx + 1}: Test positive samples: {sum(test_labels)}, Test negative samples: {len(test_labels) - sum(test_labels)}")

        
#         # Create datasets and loaders
#         train_set = AssayDataset(oversampled_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#         val_set = AssayDataset(testing, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
########################################################################################
########################################################################################
        
        # Create datasets and loaders for this fold
        train_set = AssayDataset(training, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        val_set = AssayDataset(testing, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        
        # Initialize model for this fold
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold0.2_epitope_with_all_lineage_july16_data_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        
        # Train and evaluate on this fold
        _, _, metrics = train_with_frozen_antibody_and_embedding_GRU(
            model, conf, loader_train, loader_val, fold_idx, 100, 
            f'model_threshold0.2_{antibody}_fold{fold_idx}', FINAL_MODEL, 
            save_model=False,  # Disable saving during training
            log_every_epoch=True
        )
        
        # _, _, metrics, _, _, _, _ = train_with_frozen_embeddings_only(
        #     model, conf, loader_train, loader_val, fold_idx, 100, 
        #     f'model_threshold0.2_{antibody}_fold{fold_idx}', FINAL_MODEL, 
        #     save_model=False,  # Disable saving during training
        #     log_every_epoch=True
        # )        
        print(f"Fold {fold_idx + 1} results for {antibody}:")
        print(f"MCC: {metrics[1]}\nAUC: {metrics[2]}\nACC: {metrics[0]}\n")
        all_metrics.append(metrics)
        
        # Store model and metrics for this fold
        if repeat_idx not in repeat_models:
            repeat_models[repeat_idx] = []
            repeat_metrics[repeat_idx] = []
        
        repeat_models[repeat_idx].append({
            'fold_idx': fold_idx,
            'model_state': model.state_dict(),
            'metrics': metrics
        })
        repeat_metrics[repeat_idx].append(metrics)
    
    # Calculate average performance for each repeat and find the best repeat
    best_repeat_idx = -1
    best_repeat_avg_mcc = -1.0
    
    for repeat_idx, metrics_list in repeat_metrics.items():
        avg_mcc = sum(m[MATTHEWS_CORRELATION_COEFFICIENT] for m in metrics_list) / len(metrics_list)
        if avg_mcc > best_repeat_avg_mcc:
            best_repeat_avg_mcc = avg_mcc
            best_repeat_idx = repeat_idx
    
    print(f"\nBest repeat: {best_repeat_idx + 1} with average MCC: {best_repeat_avg_mcc:.4f}")
    
    # Save all models from the best repeat
    if best_repeat_idx != -1 and best_repeat_idx in repeat_models:
        for fold_info in repeat_models[best_repeat_idx]:
            t.save({'model': fold_info['model_state']}, 
                  os.path.join(FINAL_MODEL, f'model_threshold0.2_epitope_with_all_lineage_corrected_july16_data_{antibody}_fold{fold_info["fold_idx"] + 1}_repeat{best_repeat_idx + 1}.tar'))
            print(f"Saved model from fold {fold_info['fold_idx'] + 1} of best repeat {best_repeat_idx + 1} (MCC: {fold_info['metrics'][MATTHEWS_CORRELATION_COEFFICIENT]:.4f})")
    
    # Calculate and return average metrics across all folds
    avg_metrics = [sum(m[i] for m in all_metrics)/len(all_metrics) for i in range(len(all_metrics[0]))]
    print(f"\nAverage across all folds for {antibody}:")
    print(f"MCC: {avg_metrics[1]}\nAUC: {avg_metrics[2]}\nACC: {avg_metrics[0]}\n")
    
    return avg_metrics, repeat_models.get(best_repeat_idx, [])
    

def train_final_model_5folds_1_repeat(antibody, splits_file, conf):
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)
    cross_validation_folds = all_splits[antibody]['cross_validation']
    
    for fold_idx, fold in enumerate(cross_validation_folds):
        # Only process first 5 folds of repeat 1
        if fold_idx >= 5:
            continue
            
        print(f"\nProcessing fold {fold_idx + 1}/5 (Repeat 1)")
        
        # Get train and test IDs for this specific fold
        train_ids = set(fold['train'])
        test_ids = set(fold['test'])
        
        # Filter catnap for this fold's train/test sets
        training = [a for a in catnap if a[0] in train_ids]
        testing = [a for a in catnap if a[0] in test_ids]
        
        print(f"Number of training instances in this fold: {len(training)}")
        print(f"Number of testing instances in this fold: {len(testing)}")

        # Check for single-class test set
        test_labels = [a[3] for a in testing]
        if len(set(test_labels)) < 2:
            print(f"Fold {fold_idx + 1}: Skipping due to single-class test labels")
            continue

        # Create datasets and loaders for this fold
        train_set = AssayDataset(training, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        val_set = AssayDataset(testing, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        
        # Initialize model for this fold
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_july16_data_VRC07-523-W54-LS.v3_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        
        # Train and evaluate on this fold
        # _, _, metrics = train_with_frozen_antibody_and_embedding_GRU(
        #     model, conf, loader_train, loader_val, fold_idx, 100, 
        #     f'model_threshold0.2_{antibody}_fold{fold_idx}', FINAL_MODEL, 
        #     save_model=False,
        #     log_every_epoch=True
        # )

        _, _, metrics, _, _, _, _ = train_with_frozen_embeddings_only(
            model, conf, loader_train, loader_val, fold_idx, 100, 
            f'model_threshold0.2_{antibody}_fold{fold_idx}', FINAL_MODEL, 
            save_model=False,  # Disable saving during training
            log_every_epoch=True
        )    
        
        print(f"Fold {fold_idx + 1} results for {antibody}:")
        print(f"MCC: {metrics[1]}\nAUC: {metrics[2]}\nACC: {metrics[0]}\n")
        
        # Save model
        t.save({'model': model.state_dict()}, 
              os.path.join(FINAL_MODEL, f'model_threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_july16_data_VRC07-523-W54-LS.v3_fold{fold_idx + 1}_repeat1.tar'))
        print(f"Saved model from fold {fold_idx + 1} of repeat 1")
    
    print(f"\nCompleted processing first 5 folds of repeat 1 for {antibody}")
    
    return None, None
    
def test_optimized_antibodies(experiment_name, tags = None, model_trial_name = '',
        freeze_mode = FREEZE_EMBEDDINGS_ONLY, pretrain_epochs = None, splits_file = COMPARE_SPLITS_FOR_RAWI):
    setup_logging()
    experiment_name += f' {model_trial_name}'
    experiment_id = get_experiment(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, tags = tags):
        acc, mcc, auc = [], [], []
        for antibody in ANTIBODIES_LIST:
            # cv_mean_acc, cv_mean_mcc, cv_mean_auc = test_optimized_antibody(antibody, splits_file, model_trial_name, freeze_mode, pretrain_epochs)
            cv_mean_acc, cv_mean_mcc, cv_mean_auc = test_optimized_antibody_final_model(antibody, splits_file, model_trial_name, freeze_mode, pretrain_epochs)
            acc.append(cv_mean_acc)
            mcc.append(cv_mean_mcc)
            auc.append(cv_mean_auc)
        global_acc = statistics.mean(acc)
        global_mcc = statistics.mean(mcc)
        global_auc = statistics.mean(auc)
        logging.info(f'Global ACC {global_acc}')
        logging.info(f'Global MCC {global_mcc}')
        logging.info(f'Global AUC {global_auc}')
        mlflow.log_metrics({ 'global_acc': global_acc, 'global_mcc': global_mcc, 'global_auc': global_auc })
    dump_json({'finished': 'true'}, f'finished_{antibody}_threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_final_model_5folds_1_repeat.json')

if __name__ == '__main__':
    tags = {
        # 'freeze': 'antb and embed',
        'freeze': 'embed',
        'trial': '330',
        'validation': 'uniform',
        'prune': 'treshold 0.05',
        'pretrain_epochs': 10
    }
    test_optimized_antibodies('GRU', tags = tags, model_trial_name = 'uniform_threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_final_model_5folds_1_repeat', pretrain_epochs = 100)