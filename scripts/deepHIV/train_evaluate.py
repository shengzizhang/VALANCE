import sys
sys.path.append('/home/yujieq/work/ML_training/deep_hiv_ab_pred')

import statistics
from deep_hiv_ab_pred.util.tools import read_json_file, get_experiment
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL, FREEZE_EMBEDDINGS_ONLY
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF, FINAL_MODEL
import torch as t
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import parse_catnap_sequences_to_embeddings
from deep_hiv_ab_pred.model.FC_GRU_ATT import get_FC_GRU_ATT_model
from deep_hiv_ab_pred.training.training import train_network, eval_network, eval_network_pred, train_with_frozen_antibody_and_embedding, train_with_frozen_antibody_and_embedding_predict, train_with_frozen_antibody_and_embedding_focal_loss, train_with_frozen_embeddings_only
from os.path import join
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
import mlflow
import optuna
from deep_hiv_ab_pred.util.metrics import log_metrics_per_cv_antibody
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import glob

PRETRAINING = 'pretraining'
CV = 'cross_validation'
TRAIN = 'train'
TEST = 'test'


def pretrain_net(antibody, splits_pretraining, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, pretrain_epochs):
    pretraining_assays = [a for a in catnap if a[0] in splits_pretraining]
    # rest_assays = [a for a in catnap if a[0] not in splits_pretraining]
    pretraining_assays, rest_assays = train_test_split(pretraining_assays, test_size=0.1, random_state=42)
    # assert len(pretraining_assays) == len(splits_pretraining)
    pretrain_set = AssayDataset(pretraining_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    val_set = AssayDataset(rest_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_pretrain = t.utils.data.DataLoader(pretrain_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    model = get_FC_GRU_ATT_model(conf)
    assert pretrain_epochs
    metrics_train_per_epochs, metrics_test_per_epochs, best = train_network(
        model, conf, loader_pretrain, loader_val, None, pretrain_epochs, f'model_threshold50_removed_outliers_duplicates_geomean_10E8_lineage_forCV_july16_data_{antibody}_pretrain', MODELS_FOLDER
    )
    return metrics_train_per_epochs, metrics_test_per_epochs, best




def load_pretrain_net(antibody, splits_pretraining, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, pretrain_epochs):
    # Load the pretrained model directly - we should do this for focal loss/ROS experiments as we want the pretrained process to be fixed/the same
    # checkpoint_path = os.path.join(MODELS_FOLDER, f'publication_ready/model_threshold50_july16_data_{antibody}_pretrain.tar')
    checkpoint_path = os.path.join(MODELS_FOLDER, f'model_threshold50_epitope_with_all_lineage_removed_outliers_geomean_july16_data_{antibody}_pretrain.tar')
    print(f"Loading pretrained model from {checkpoint_path}")
    model = get_FC_GRU_ATT_model(conf)
    checkpoint = t.load(checkpoint_path)
    # Ensure the model state is loaded correctly
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Model state loaded successfully.")
    else:
        raise KeyError(f"'model' key not found in checkpoint at {checkpoint_path}.")
    # Retrieve metrics and best values
    metrics_train_per_epochs = checkpoint.get('metrics_train_per_epochs', [])
    metrics_test_per_epochs = checkpoint.get('metrics_test_per_epochs', [])
    best = checkpoint.get('best', None)

    # Fallback for missing metrics in older checkpoints
    if not metrics_train_per_epochs or not metrics_test_per_epochs:
        print("Warning: metrics not found in checkpoint. Initializing as empty.")
        metrics_train_per_epochs, metrics_test_per_epochs = [], []

    print(f"Training metrics (per epoch): {metrics_train_per_epochs}")
    print(f"Validation metrics (per epoch): {metrics_test_per_epochs}")
    print(f"Best metrics: {best}")

    return model, metrics_train_per_epochs, metrics_test_per_epochs, best




def pseudo_pretrain_net(antibody, splits_pretraining, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, pretrain_epochs):
    # Prepare file path for the checkpoint
    checkpoint_path = os.path.join(MODELS_FOLDER, f'model_threshold50_Focal_Loss_july16_data_{antibody}_pretrain.tar')
    
    # Check if the checkpoint already exists to avoid overwriting
    if os.path.exists(checkpoint_path):
        print(f"Pretrained checkpoint already exists at {checkpoint_path}. Skipping pretraining.")
        return None, None, None  # No need to proceed further
    
    # Generate dummy pretraining data (optional, if you need to log or use it elsewhere)
    pretraining_assays = [a for a in catnap if a[0] in splits_pretraining]
    # Splitting data for pretraining and validation (though not used in dummy case)
    pretraining_assays, rest_assays = train_test_split(pretraining_assays, test_size=0.1, random_state=42)
    
    # Prepare a dummy dataset and dataloader (not used for training, but kept for consistency)
    pretrain_set = AssayDataset(pretraining_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_pretrain = t.utils.data.DataLoader(pretrain_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
    
    # Create a model with the given configuration
    model = get_FC_GRU_ATT_model(conf)
    
    # Create a dummy checkpoint
    dummy_checkpoint = {
        'model': model.state_dict(),  # Save the initialized model state
        'optimizer': None,  # Placeholder for optimizer state (not used)
        'epoch': 0,  # Indicate that no pretraining has occurred
        'metrics': {},  # Empty metrics since no training was performed
    }
    
    # Save the dummy checkpoint
    t.save(dummy_checkpoint, checkpoint_path)
    print(f"Dummy checkpoint saved at {checkpoint_path}")
    
    # Return placeholder values for metrics and best state
    return [], [], None

# def cross_validate_antibody(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
#     antibody_heavy_seq, trial = None, cv_folds_trim = 100, cv_folds_skip = 0, freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS):

#     cv_metrics = []
#     for (i, cv_fold) in enumerate(splits_cv[cv_folds_skip : cv_folds_skip + cv_folds_trim]):
#         train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
#         train_assays = [a for a in catnap if a[0] in train_ids]
#         test_assays = [a for a in catnap if a[0] in test_ids]
#         assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids)
#         train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#         test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#         loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
#         loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle = False, collate_fn = zero_padding, num_workers = 0)
#         model = get_FC_GRU_ATT_model(conf)
#         checkpoint = t.load(join(MODELS_FOLDER, f'model_our_data_{antibody}_pretrain.tar'))
#         model.load_state_dict(checkpoint['model'])
#         if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
#             _, _, metrics = train_with_frozen_antibody_and_embedding(
#                 model, conf, loader_train, loader_test, i, 100, f'model_our_data_{antibody}', MODELS_FOLDER, False, log_every_epoch = False
#             )
#         elif freeze_mode == FREEZE_ALL:
#             metrics = eval_network(model, loader_test)
#         else:
#             raise 'Must provide a freeze mode.'
#         cv_metrics.append(metrics)
#         if trial:
#             trial.report(metrics[MATTHEWS_CORRELATION_COEFFICIENT], i)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()
#     return cv_metrics


# def cross_validate_antibody(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
#                             antibody_heavy_seq, trial=None, cv_folds_trim=100, cv_folds_skip=0, freeze_mode=FREEZE_ANTIBODY_AND_EMBEDDINGS):
#     cv_metrics = []
#     train_mcc_CVs, val_mcc_CVs = [], []
#     train_losses, val_losses = [], []
#     for (i, cv_fold) in enumerate(splits_cv[cv_folds_skip: cv_folds_skip + cv_folds_trim]):
#         train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
#         train_assays = [a for a in catnap if a[0] in train_ids]
#         test_assays = [a for a in catnap if a[0] in test_ids]


#         # Debugging prints
#         missing_train_ids = set(train_ids) - set(a[0] for a in train_assays)
#         missing_test_ids = set(test_ids) - set(a[0] for a in test_assays)

#         print(f"Fold {i + 1}:")
#         print(f"  Expected train IDs: {len(train_ids)}, Found train assays: {len(train_assays)}")
#         print(f"  Expected test IDs: {len(test_ids)}, Found test assays: {len(test_assays)}")
#         print(f"  Missing train IDs: {len(missing_train_ids)}")
#         print(f"  Missing test IDs: {len(missing_test_ids)}")
#         if missing_train_ids:
#             print(f"  Train IDs missing in assays: {missing_train_ids}")
#         if missing_test_ids:
#             print(f"  Test IDs missing in assays: {missing_test_ids}")

#         assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids), (
#             f"Assertion failed in fold {i + 1}: "
#             f"Expected {len(train_ids)} train IDs but found {len(train_assays)} train assays. "
#             f"Expected {len(test_ids)} test IDs but found {len(test_assays)} test assays."
#         )

    
#         # assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids)

        
#         train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#         test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#         loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
#         loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle=False, collate_fn=zero_padding, num_workers=0)
#         model = get_FC_GRU_ATT_model(conf)
#         checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold50_july16_data_{antibody}_pretrain.tar'))
#         model.load_state_dict(checkpoint['model'])
#         if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
#             _, _, metrics, train_mcc, val_mcc, train_loss, val_loss = train_with_frozen_antibody_and_embedding(
#                 model, conf, loader_train, loader_test, i, 100, f'model_threshold50_july16_data_{antibody}', MODELS_FOLDER, False, log_every_epoch=False
#             )
#         elif freeze_mode == FREEZE_ALL:
#             metrics = eval_network(model, loader_test)
#         else:
#             raise 'Must provide a freeze mode.'

#         cv_metrics.append(metrics)
#         train_mcc_CVs.append(train_mcc)
#         val_mcc_CVs.append(val_mcc)
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         if trial:
#             trial.report(metrics[MATTHEWS_CORRELATION_COEFFICIENT], i)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()
#     return cv_metrics, train_mcc_CVs, val_mcc_CVs, train_losses, val_losses

def cross_validate_antibody(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
                            antibody_heavy_seq, trial=None, cv_folds_trim=100, cv_folds_skip=0, freeze_mode=FREEZE_EMBEDDINGS_ONLY):
    cv_metrics = []
    train_mcc_CVs, val_mcc_CVs = [], []
    train_losses, val_losses = [], []
    
    catnap_ids = set([a[0] for a in catnap])  # Set of all IDs in catnap
    
    for (i, cv_fold) in enumerate(splits_cv[cv_folds_skip: cv_folds_skip + cv_folds_trim]):
        train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
        
        # Ensure that all train/test IDs exist in catnap
        missing_train_ids = set(train_ids) - catnap_ids
        missing_test_ids = set(test_ids) - catnap_ids
        
        if missing_train_ids or missing_test_ids:
            print(f"Fold {i + 1}: Missing train IDs: {missing_train_ids}, Missing test IDs: {missing_test_ids}")
            continue  # Skip this fold if there are missing IDs
        
        train_assays = [a for a in catnap if a[0] in train_ids]
        test_assays = [a for a in catnap if a[0] in test_ids]

        print(f"Fold {i + 1}:")
        print(f"  Expected train IDs: {len(train_ids)}, Found train assays: {len(train_assays)}")
        print(f"  Expected test IDs: {len(test_ids)}, Found test assays: {len(test_assays)}")

        assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids), (
            f"Assertion failed in fold {i + 1}: "
            f"Expected {len(train_ids)} train IDs but found {len(train_assays)} train assays. "
            f"Expected {len(test_ids)} test IDs but found {len(test_assays)} test assays."
        )

        train_labels = [a[3] for a in train_assays]  # Assuming ground truth is at index 3
        test_labels = [a[3] for a in test_assays]
        print(f"Fold {i + 1}: Train positive samples: {sum(train_labels)}, Train negative samples: {len(train_labels) - sum(train_labels)}")
        print(f"Fold {i + 1}: Test positive samples: {sum(test_labels)}, Test negative samples: {len(test_labels) - sum(test_labels)}")


        # Skip fold if the test set has only one class
        if len(set(test_labels)) < 2:
            print(f"Fold {i + 1}: Skipping due to single-class test labels.")
            continue
            
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)

        # # Filter virus_seq and virus_pngs_mask to exclude records that start with ">C.ZM.x.ZM176_66." but include the exact match
        # prefix_to_exclude = "ZM176_66"
        
        # # Print original keys for virus_seq
        # print("Original virus_seq keys:", list(virus_seq.keys()))
        # print("Original virus_pngs_mask keys:", list(virus_pngs_mask.keys()))
        
        # # Filter virus_seq
        # filtered_virus_seq = {key: value for key, value in virus_seq.items() 
        #                       if not key.startswith(prefix_to_exclude) or key == prefix_to_exclude}
        
        # # Filter virus_pngs_mask
        # filtered_virus_pngs_mask = {key: value for key, value in virus_pngs_mask.items() 
        #                             if not key.startswith(prefix_to_exclude) or key == prefix_to_exclude}
        
        # # Print filtered keys for virus_seq
        # print("Filtered virus_seq keys:", list(filtered_virus_seq.keys()))
        # print("Filtered virus_pngs_mask keys:", list(filtered_virus_pngs_mask.keys()))
        
        # # Print out what was excluded
        # excluded_virus_seq = {key: value for key, value in virus_seq.items() if key.startswith(prefix_to_exclude) and key != prefix_to_exclude}
        # excluded_virus_pngs_mask = {key: value for key, value in virus_pngs_mask.items() if key.startswith(prefix_to_exclude) and key != prefix_to_exclude}
        
        # print("Excluded virus_seq keys:", list(excluded_virus_seq.keys()))
        # print("Excluded virus_pngs_mask keys:", list(excluded_virus_pngs_mask.keys()))
        
        # # Pass the filtered dictionaries to the test_set
        # test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, filtered_virus_seq, filtered_virus_pngs_mask)
        
        

        # # Check if the key 'ZM176_66' exists in virus_seq
        # key_to_check = '093_LN_SGA9'
        
        # if key_to_check in virus_seq:
        #     # Print the number of values associated with this key
        #     print(f"Number of values for key '{key_to_check}' in virus_seq: {len(virus_seq[key_to_check])}")
        # else:
        #     print(f"Key '{key_to_check}' not found in virus_seq.")
        
                
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle=False, collate_fn=zero_padding, num_workers=0)
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold50_removed_outliers_duplicates_geomean_10E8_lineage_forCV_july16_data_{antibody}_pretrain.tar'))
        # checkpoint = t.load(join(MODELS_FOLDER, f'publication_ready/model_threshold50_july16_data_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        

        if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
            _, _, metrics, train_mcc, val_mcc, train_loss, val_loss = train_with_frozen_antibody_and_embedding(
                model, conf, loader_train, loader_test, i, 100,
                f'model_threshold50_removed_outliers_duplicates_geomean_epitope_with_all_lineage_july16_data_{antibody}',
                MODELS_FOLDER, False, log_every_epoch=False
            )
        elif freeze_mode == FREEZE_EMBEDDINGS_ONLY:
            _, _, metrics, train_mcc, val_mcc, train_loss, val_loss = train_with_frozen_embeddings_only(
                model, conf, loader_train, loader_test, i, 100,
                f'model_threshold50_removed_outliers_duplicates_geomean_10E8_lineage_forCV_july16_data_{antibody}',
                MODELS_FOLDER, False, log_every_epoch=False
            )
        elif freeze_mode == FREEZE_ALL:
            metrics = eval_network(model, loader_test)
        else:
            raise ValueError('Must provide a proper freeze mode.')

        
        cv_metrics.append(metrics)
        train_mcc_CVs.append(train_mcc)
        val_mcc_CVs.append(val_mcc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if trial:
            trial.report(metrics[MATTHEWS_CORRELATION_COEFFICIENT], i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return cv_metrics, train_mcc_CVs, val_mcc_CVs, train_losses, val_losses

###########################################################
####### Updating cross_validate_antibody so that for each CV fold (from the specified slice of splits_cv), we retrieve the predictions from the test set and—using the fact that the test DataLoader (with shuffle=False) preserves the same order as your test assays—we build a list of dictionaries that record:
# The CV fold (as a number)
# The sample’s ID (from a[0])
# The antibody (from a[1])
# The virus (from a[2])
# The prediction (from the model)
# The corresponding ground truth
# Finally, after processing all folds, the code concatenates the records from all folds and saves them to a CSV file (here named "cv_predictions_results.csv").


import pandas as pd

def cross_validate_antibody_predict(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
                            antibody_heavy_seq, trial=None, cv_folds_trim=100, cv_folds_skip=0, 
                            freeze_mode=FREEZE_ANTIBODY_AND_EMBEDDINGS):
    cv_metrics = []
    train_mcc_CVs, val_mcc_CVs = [], []
    train_losses, val_losses = [], []
    
    # List to store prediction details across all CV folds
    cv_results = []  
    
    catnap_ids = set([a[0] for a in catnap])  # Set of all IDs in catnap
    
    # Loop over the specified CV folds
    for (i, cv_fold) in enumerate(splits_cv[cv_folds_skip: cv_folds_skip + cv_folds_trim]):
        train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
        
        # Ensure that all train/test IDs exist in catnap
        missing_train_ids = set(train_ids) - catnap_ids
        missing_test_ids = set(test_ids) - catnap_ids
        
        if missing_train_ids or missing_test_ids:
            print(f"Fold {i + 1}: Missing train IDs: {missing_train_ids}, Missing test IDs: {missing_test_ids}")
            continue  # Skip this fold if there are missing IDs
        
        train_assays = [a for a in catnap if a[0] in train_ids]
        test_assays = [a for a in catnap if a[0] in test_ids]

        print(f"Fold {i + 1}:")
        print(f"  Expected train IDs: {len(train_ids)}, Found train assays: {len(train_assays)}")
        print(f"  Expected test IDs: {len(test_ids)}, Found test assays: {len(test_assays)}")

        assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids), (
            f"Assertion failed in fold {i + 1}: "
            f"Expected {len(train_ids)} train IDs but found {len(train_assays)} train assays. "
            f"Expected {len(test_ids)} test IDs but found {len(test_assays)} test assays."
        )

        train_labels = [a[3] for a in train_assays]  # Assuming ground truth is at index 3
        test_labels = [a[3] for a in test_assays]
        print(f"Fold {i + 1}: Train positive samples: {sum(train_labels)}, Train negative samples: {len(train_labels) - sum(train_labels)}")
        print(f"Fold {i + 1}: Test positive samples: {sum(test_labels)}, Test negative samples: {len(test_labels) - sum(test_labels)}")

        # Skip fold if the test set has only one class
        if len(set(test_labels)) < 2:
            print(f"Fold {i + 1}: Skipping due to single-class test labels.")
            continue
            
        # Create datasets and loaders for this fold
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle=False, collate_fn=zero_padding, num_workers=0)
        
        # Reload the model and its checkpoint for this fold
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold50_epitope_with_all_lineage_panel_july16_data_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        
        # --- Training/Evaluation Branch ---
        if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
            _, _, metrics, train_mcc, val_mcc, train_loss, val_loss, fold_predictions, fold_ground_truths = train_with_frozen_antibody_and_embedding_predict(
                model, conf, loader_train, loader_test, i, 100, f'model_threshold50_epitope_with_all_lineage_panel_july16_data_{antibody}', MODELS_FOLDER, False, log_every_epoch=False
            )     
        elif freeze_mode == FREEZE_ALL:
            metrics = eval_network(model, loader_test)
        else:
            raise 'Must provide a freeze mode.'
        
        cv_metrics.append(metrics)
        train_mcc_CVs.append(train_mcc)
        val_mcc_CVs.append(val_mcc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # For each sample in the test set, record its cv fold, id, antibody, virus, prediction, and ground truth
        for j, assay in enumerate(test_assays):
            cv_results.append({
                "cv_fold": i + cv_folds_skip + 1,  # 1-indexed fold number
                "id": assay[0],
                "antibody": assay[1],
                "virus": assay[2],
                "prediction": fold_predictions[j],
                "ground_truth": fold_ground_truths[j]
            })
        
        if trial:
            trial.report(metrics[MATTHEWS_CORRELATION_COEFFICIENT], i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # After processing all folds, save the concatenated results to a CSV file
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(f"/home/yujieq/work/ML_training/deep_hiv_ab_pred/{antibody}_cv_predictions_results_threshold50_panel.csv", index=False)
    print("CV predictions and ground truths saved to cv_predictions_results.csv")
    
    return cv_metrics, train_mcc_CVs, val_mcc_CVs, train_losses, val_losses

###########################################################
####### Updating cross_validate_antibody for Focal Loss/ROS; 
##Note that we will skip pretrain process and load the pretrained model directly with "load_pretrain_net", OR, you DON'T have to load pretrained model at all since the below function itself will load the saved pretrained model. However, we should still do the 1000 trials but with either "cross_validate_antibody_focal_loss" / "cross_validate_antibody_with_oversampler"
def cross_validate_antibody_focal_loss(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
                            antibody_heavy_seq, trial=None, cv_folds_trim=100, cv_folds_skip=0, freeze_mode=FREEZE_ANTIBODY_AND_EMBEDDINGS):
    cv_metrics = []
    train_mcc_CVs, val_mcc_CVs = [], []
    train_losses, val_losses = [], []
    
    catnap_ids = set([a[0] for a in catnap])  # Set of all IDs in catnap
    
    for (i, cv_fold) in enumerate(splits_cv[cv_folds_skip: cv_folds_skip + cv_folds_trim]):
        train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
        
        # Ensure that all train/test IDs exist in catnap
        missing_train_ids = set(train_ids) - catnap_ids
        missing_test_ids = set(test_ids) - catnap_ids
        
        if missing_train_ids or missing_test_ids:
            print(f"Fold {i + 1}: Missing train IDs: {missing_train_ids}, Missing test IDs: {missing_test_ids}")
            continue  # Skip this fold if there are missing IDs
        
        train_assays = [a for a in catnap if a[0] in train_ids]
        test_assays = [a for a in catnap if a[0] in test_ids]

        print(f"Fold {i + 1}:")
        print(f"  Expected train IDs: {len(train_ids)}, Found train assays: {len(train_assays)}")
        print(f"  Expected test IDs: {len(test_ids)}, Found test assays: {len(test_assays)}")

        assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids), (
            f"Assertion failed in fold {i + 1}: "
            f"Expected {len(train_ids)} train IDs but found {len(train_assays)} train assays. "
            f"Expected {len(test_ids)} test IDs but found {len(test_assays)} test assays."
        )

        train_labels = [a[3] for a in train_assays]  # Assuming ground truth is at index 3
        test_labels = [a[3] for a in test_assays]
        print(f"Fold {i + 1}: Train positive samples: {sum(train_labels)}, Train negative samples: {len(train_labels) - sum(train_labels)}")
        print(f"Fold {i + 1}: Test positive samples: {sum(test_labels)}, Test negative samples: {len(test_labels) - sum(test_labels)}")


        # Skip fold if the test set has only one class
        if len(set(test_labels)) < 2:
            print(f"Fold {i + 1}: Skipping due to single-class test labels.")
            continue
            
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
               
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle=False, collate_fn=zero_padding, num_workers=0)
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold50_epitope_with_all_lineage_removed_outliers_geomean_july16_data_{antibody}_pretrain.tar'))
        # checkpoint = t.load(join(MODELS_FOLDER, f'publication_ready/model_threshold50_july16_data_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        
        if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
            _, _, metrics, train_mcc, val_mcc, train_loss, val_loss = train_with_frozen_antibody_and_embedding_focal_loss(
                model, conf, loader_train, loader_test, i, 100, f'model_threshold50_epitope_with_all_lineage_removed_outliers_geomean_july16_data_Focal_Loss_{antibody}', MODELS_FOLDER, False, log_every_epoch=False
            )            
        elif freeze_mode == FREEZE_ALL:
            metrics = eval_network(model, loader_test)
        else:
            raise 'Must provide a freeze mode.'

        cv_metrics.append(metrics)
        train_mcc_CVs.append(train_mcc)
        val_mcc_CVs.append(val_mcc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if trial:
            trial.report(metrics[MATTHEWS_CORRELATION_COEFFICIENT], i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return cv_metrics, train_mcc_CVs, val_mcc_CVs, train_losses, val_losses


def cross_validate_antibody_with_oversampler(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
                                             antibody_heavy_seq, trial=None, cv_folds_trim=100, cv_folds_skip=0,
                                             freeze_mode=FREEZE_ANTIBODY_AND_EMBEDDINGS):
    cv_metrics = []
    train_mcc_CVs, val_mcc_CVs = [], []
    train_losses, val_losses = [], []

    catnap_ids = set([a[0] for a in catnap])  # Set of all IDs in catnap

    for (i, cv_fold) in enumerate(splits_cv[cv_folds_skip: cv_folds_skip + cv_folds_trim]):
        train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]

        # Ensure that all train/test IDs exist in catnap
        train_assays = [a for a in catnap if a[0] in train_ids]
        test_assays = [a for a in catnap if a[0] in test_ids]

        train_labels = [a[3] for a in train_assays]  # Assuming ground truth is at index 3
        test_labels = [a[3] for a in test_assays]

        # Skip fold if the test set has only one class
        if len(set(test_labels)) < 2:
            print(f"Fold {i + 1}: Skipping due to single-class test labels.")
            continue

        # Prepare features and labels for oversampling
        train_features = [a[:3] for a in train_assays]  # Extract features (antibody, virus, etc.)
        train_labels = [int(a[3]) for a in train_assays]  # Ensure labels are integers (0/1)

        # Apply RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        oversampled_features, oversampled_labels = ros.fit_resample(train_features, train_labels)

        # Rebuild the oversampled dataset
        oversampled_assays = [
            (*features, label) for features, label in zip(oversampled_features, oversampled_labels)
        ]

        print(f"Fold {i + 1}: Train positive samples: {sum(train_labels)}, Train negative samples: {len(train_labels) - sum(train_labels)}")
        print(f"Fold {i + 1}: Oversampled Train positive samples: {sum(oversampled_labels)}, Train negative samples: {len(oversampled_labels) - sum(oversampled_labels)}")
        print(f"Fold {i + 1}: Test positive samples: {sum(test_labels)}, Test negative samples: {len(test_labels) - sum(test_labels)}")

        
        # Create datasets and loaders
        train_set = AssayDataset(oversampled_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)

        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle=True, collate_fn=zero_padding, num_workers=0)
        loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle=False, collate_fn=zero_padding, num_workers=0)

        model = get_FC_GRU_ATT_model(conf)
        # checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold50_ROS_july16_data_{antibody}_pretrain.tar'))
        checkpoint = t.load(join(MODELS_FOLDER, f'model_threshold50_epitope_with_all_lineage_removed_outliers_geomean_july16_data_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])

        if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
        #     _, _, metrics, train_mcc, val_mcc, train_loss, val_loss = train_with_frozen_antibody_and_embedding(
        #         model, conf, loader_train, loader_test, i, 100, f'model_threshold50_ROS&FocalLoss_july16_data_{antibody}', MODELS_FOLDER, False, log_every_epoch=False
        #     )
            _, _, metrics, train_mcc, val_mcc, train_loss, val_loss = train_with_frozen_antibody_and_embedding(
                model, conf, loader_train, loader_test, i, 100, f'model_threshold50_epitope_with_all_lineage_removed_outliers_geomean_july16_data_ROS_{antibody}', MODELS_FOLDER, False, log_every_epoch=False
            )   
        elif freeze_mode == FREEZE_ALL:
            metrics = eval_network(model, loader_test)
        else:
            raise 'Must provide a freeze mode.'

        cv_metrics.append(metrics)
        train_mcc_CVs.append(train_mcc)
        val_mcc_CVs.append(val_mcc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if trial:
            trial.report(metrics[MATTHEWS_CORRELATION_COEFFICIENT], i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return cv_metrics, train_mcc_CVs, val_mcc_CVs, train_losses, val_losses



def train_net(experiment_name, tags = None, freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS):
    experiment_id = get_experiment(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, tags = tags):
        conf = read_json_file(DEFAULT_CONF)
        mlflow.log_artifact(DEFAULT_CONF, 'base_conf.json')
        all_splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)
        # mlflow.log_artifact(COMPARE_SPLITS_FOR_RAWI)
        catnap = read_json_file(CATNAP_FLAT)
        # mlflow.log_artifact(CATNAP_FLAT)
        virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
            conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS']
        )
        acc, mcc = [], []
        for i, (antibody, splits) in enumerate(all_splits.items()):
            logging.info(f'{i}. Antibody {antibody}')
            pretrain_net(antibody, splits[PRETRAINING], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)
            cv_metrics = cross_validate_antibody(antibody, splits[CV], catnap, conf, virus_seq, virus_pngs_mask,
                antibody_light_seq, antibody_heavy_seq, freeze_mode = freeze_mode)
            cv_mean_acc, cv_mean_mcc = log_metrics_per_cv_antibody(cv_metrics, antibody)
            acc.append(cv_mean_acc)
            mcc.append(cv_mean_mcc)
        global_acc = statistics.mean(acc)
        global_mcc = statistics.mean(mcc)
        logging.info('Global ACC ' + global_acc)
        logging.info('Global MCC ' + global_mcc)
        mlflow.log_metrics({ 'global_acc': global_acc, 'global_mcc': global_mcc })


# def eval_newdata(antibody, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, conf):
#         test_assays = []
#         j=0
#         for i in virus_seq:
#             test_assays.append([j,antibody,i,False])
#             j+=1
#         #print(virus_pngs_mask)
#         test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
#         loader_test = t.utils.data.DataLoader(test_set, 1, shuffle = False, collate_fn = zero_padding, num_workers = 0)
#         model = get_FC_GRU_ATT_model(conf)
#         # checkpoint = t.load(join(FINAL_MODEL, f'model_{antibody}_GRU_cv1.tar'))
#         checkpoint = t.load(f'/home/yujieq/work/ML_training/deep_hiv_ab_pred/deep_hiv_ab_pred/final_model/model_threshold50_random800_clade_seed33_{antibody}_GRU_cv1.tar')
#         model.load_state_dict(checkpoint['model'])
#         preds, ground_tru = eval_network_pred(model, loader_test)
#         j=0
#         state='sensitive'
#         for i in virus_seq:
#             if preds[j]>0.5:
#                 state='sensitive'
#             else:
#                 state='resistant'
#             print(f'{i}\t{preds[j]}\t{ground_tru[j]}\t{state}\n')
#             j+=1


def eval_newdata(antibody, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, conf, threshold):
    test_assays = []
    j = 0
    for seq_id in virus_seq:
        test_assays.append([j, antibody, seq_id, False])  # [index, antibody, virus_id, ground_truth]
        j += 1
    
    test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_test = t.utils.data.DataLoader(test_set, 1, shuffle=False, collate_fn=zero_padding, num_workers=0)
    
    model = get_FC_GRU_ATT_model(conf)
    
    # Find the best model file for this antibody and threshold
    model_pattern = f'model_threshold{threshold}_epitope_with_all_lineage_removed_outliers_geomean_july16_data_{antibody}_best_fold*.tar'
    model_files = glob.glob(os.path.join(FINAL_MODEL, model_pattern))
    
    if not model_files:
        raise FileNotFoundError(
            f"No best fold model found for antibody {antibody} and threshold {threshold}\n"
            f"Looked for pattern: {model_pattern} in {FINAL_MODEL}"
        )
    
    # Sort by modification time and take the newest if multiple exist
    model_files.sort(key=os.path.getmtime, reverse=True)
    best_model_path = model_files[0]
    
    checkpoint = t.load(best_model_path)
    model.load_state_dict(checkpoint['model'])
    
    preds, ground_tru = eval_network_pred(model, loader_test)
    
    j = 0
    state='sensitive'
    for i in virus_seq:
        if preds[j]>0.5:
            state='sensitive'
        else:
            state='resistant'
        print(f'{i}\t{preds[j]}\t{ground_tru[j]}\t{state}\n')
        j+=1


if __name__ == '__main__':
    tags = {
        'note1': 'virus seq aligned unlike in ICERI2021',
        'note2': 'no parameters are freezed'
    }
    train_net('ICERI2021', tags, freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS)
