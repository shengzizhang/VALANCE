import sys
import os

sys.path.append('/home/yujieq/work/ML_training/deep_hiv_ab_pred')
import os
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, device, get_experiment
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL_BUT_LAST_LAYER, FREEZE_ALL, ANTIBODIES_LIST,HYPERPARAM_FOLDER_ANTIBODIES
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF, FINAL_MODEL
import torch as t
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT,VIRUS_FILE,VIRUS_WITH_PNGS_FILE
import re
from os.path import join
import mlflow   
import statistics
from os.path import join
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
import argparse
import os.path
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import sequence_to_indexes,pngs_mask_to_kemr_tensor, parse_catnap_sequences_to_embeddings
# from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import eval_newdata
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.model.FC_GRU_ATT import get_FC_GRU_ATT_model
from deep_hiv_ab_pred.util.tools import to_numpy
import glob
import numpy as np




def add_properties_from_base_config(conf, base_conf):
    for prop in base_conf:
        if prop not in conf:
            conf[prop] = base_conf[prop]
    return conf
        
def N_to_O(seq):
    seq1=seq.upper()
    seqnoaln=str(seq1)
    seqnoaln=seqnoaln.replace('-','')
    pos=[]
    j=0
    for i in range(len(seq1)):
        
        if seq1[i]== 'N':
            # if seqnoaln[j+1] != 'P' and re.search('[ST]',seqnoaln[j+2]):
            if j + 1 < len(seqnoaln) and seqnoaln[j+1] != 'P' and j + 2 < len(seqnoaln) and re.search('[ST]', seqnoaln[j+2]):
                #print(f"{i} {j} {seq1[i:i+4]} {seqnoaln[j:j+4]}")
                if seqnoaln[j+1] == 'N' and re.search('[ST]',seqnoaln[j+3]):
                    print(f"skip {i}")
                else:
                    seq1 = seq1[:i] + 'O' + seq1[i + 1:]
        if seq1[i] != '-':
            j+=1
            
    return seq1
    

    

def mafft_aln(virus_seq_path, mafft_path):
    """Perform MAFFT alignment on the given virus sequence file"""
    # Use the global VIRUS_FILE constant for the reference alignment
    os.system(f"{mafft_path} --thread 20 --addfull {virus_seq_path} --keeplength {VIRUS_FILE} > {virus_seq_path}_aln.fasta")    
    # Read original sequences to get IDs
    seqt = SeqIO.parse(virus_seq_path, "fasta")
    target_ids = [fasta.id for fasta in seqt]
    
    # Filter the aligned file to only include our original sequences
    with open(f"{virus_seq_path}_aln1.fasta", 'w') as f:
        for fasta in SeqIO.parse(f"{virus_seq_path}_aln.fasta", "fasta"):
            if fasta.id in target_ids:
                SeqIO.write(fasta, f, 'fasta')


def eval_network_pred(model, loader):
    model.eval()
    prediction_list, ground_truth_list = [], [] 
    with t.no_grad():
        for i, (ab_light, ab_heavy, virus, pngs_mask, ground_truth) in enumerate(loader):
            # Forward pass
            pred = model.forward(ab_light, ab_heavy, virus, pngs_mask)
            
            # Convert predictions and ground truth to numpy arrays
            pred_np = to_numpy(pred).flatten()  # Ensure it's a 1D array
            ground_truth_np = to_numpy(ground_truth).flatten()  # Ensure it's a 1D array
            
            # Append to lists
            prediction_list.append(pred_np)
            ground_truth_list.append(ground_truth_np)
            
            # # Debugging: Print shapes
            # print(f"Batch {i}: pred shape = {pred_np.shape}, ground truth shape = {ground_truth_np.shape}")

    # Concatenate all predictions and ground truths
    all_predictions = np.concatenate(prediction_list)
    all_ground_truths = np.concatenate(ground_truth_list)
    
    return all_predictions, all_ground_truths
    
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################    
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
    model_pattern = f'model_threshold{threshold}_epitope_with_all_lineage_july16_data_{antibody}_best_fold*.tar'
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

def process_best1_of100(antibody, threshold, virus_seq_file, output_dir, mafft_path):
    """Process a single antibody with given threshold"""
    print(f"\nProcessing antibody: {antibody} with threshold {threshold}")
    
    # Set up file paths
    output_file = os.path.join(output_dir, f"Predicted_threshold{threshold}_CATNAP_399_more_to_add_oct1_25_best1_of100_{antibody}.txt")
    # output_file = os.path.join(output_dir, f"Predicted_threshold{threshold}_Xueling_30_Env_SGA_PCR_360_m13_best1_of100_{antibody}.txt")
    # virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_virseqs_aa.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_virseqs_aa_384more_to_add.fasta")
    virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_virseqs_aa_399_more_to_add_oct1_25.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_virseqs_aa_xueling.fasta.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"360_m13_aa.fasta")
    
    if not os.path.exists(virus_seq_path):
        print(f"Virus sequence file not found: {virus_seq_path}")
        return
    
    # MAFFT alignment (only need to do this once)
    aligned_file = f"{virus_seq_path}_aln1.fasta"
    if not os.path.exists(aligned_file):
        mafft_aln(virus_seq_path, mafft_path)
    
    # Load configurations
    base_conf = read_json_file(DEFAULT_CONF)
    conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_epitope_with_all_lineage_{antibody}.json')
    
    if not os.path.exists(conf_file):
        print(f"Config file not found: {conf_file}")
        return
    
    conf = read_json_file(conf_file)
    conf = add_properties_from_base_config(conf, base_conf)
    
    # Process sequences
    virus_seq1, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
        base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS']
    )
    
    virus_seq = {}
    virus_seq_mask = {}
    
            
    for seq_record in SeqIO.parse(aligned_file, "fasta"):
        seq1 = str(seq_record.seq.upper())
        seq1 = seq1.replace('*','-')
        seq1 = re.sub('[BJOUZ]', 'X', seq1)
        virus_seq[seq_record.id] = sequence_to_indexes(seq1, base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'])
        binary_mask = [1. if c == 'O' else 0. for c in N_to_O(seq1)]
        virus_seq_mask[seq_record.id] = pngs_mask_to_kemr_tensor(binary_mask, base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'])
    
    # Run prediction and save output
    with open(output_file, "w") as output_file:
        original_stdout = sys.stdout
        sys.stdout = output_file
        try:
            eval_newdata(antibody, virus_seq, virus_seq_mask, antibody_light_seq, antibody_heavy_seq, conf, threshold)
        finally:
            sys.stdout = original_stdout
    
    print(f"Completed processing for {antibody}. Results saved to {output_file}")
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def process_all10folds_of_best_repeat(antibody, threshold, virus_seq_file, output_dir, mafft_path):
    # Find all model files for this antibody and threshold
    # model_pattern = f'model_threshold{threshold}_epitope_with_all_lineage_removed_outliers_duplicates_geomean_july16_data_{antibody}_fold*_repeat*.tar'
    # model_pattern = f'model_threshold{threshold}_epitope_with_all_lineage_removed_outliers_geomean_ROS_july16_data_{antibody}_fold*_repeat*.tar'
    # model_pattern = f'model_threshold{threshold}_removed_outliers_duplicates_geomean_PGT145_lineage_forCV_july16_data_PGT145_lineage_fold*_repeat*.tar'
    # model_pattern = f'model_threshold{threshold}_removed_outliers_duplicates_geomean_VRC01_01_07_Clade_forCV_july16_data_VRC01_lineage_fold*_repeat*.tar'
    model_pattern = f'model_threshold{threshold}_epitope_with_all_lineage_july16_data_{antibody}_fold*_repeat*.tar'
    model_files = glob.glob(os.path.join(FINAL_MODEL, model_pattern))
    
    print(f"Found {len(model_files)} model files for {antibody}")
    
    # Load configurations and process sequences once (common for all folds)
    base_conf = read_json_file(DEFAULT_CONF)
    # conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_epitope_with_all_lineage_removed_outliers_duplicates_geomean_{antibody}.json'
    # conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_epitope_with_all_lineage_removed_outliers_geomean_ROS_{antibody}.json')
    # conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_removed_outliers_duplicates_geomean_PGT145_lineage_forCV_PGT145_lineage.json')
    # conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_removed_outliers_duplicates_geomean_VRC01_01_07_Clade_forCV_VRC01_lineage.json')
    conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_epitope_with_all_lineage_{antibody}.json')
    
    conf = read_json_file(conf_file)
    conf = add_properties_from_base_config(conf, base_conf)

    #####################################
    # Process sequences once
    # virus_seq_path = os.path.join(virus_seq_file, f"Xueling_virseqs_aa.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_virseqs_aa_384more_to_add.fasta")
    virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_virseqs_aa_399_more_to_add_oct1_25.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"virseqs_aa_oct1_25.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"VRC01_viruses_seqs_from_399_more_to_add_oct1_25_identity0.8_representatives.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"414_m9_aa.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"LAT-03-064_RNA_demuxed_FAD_filtered_longest_orfs_aa.fasta")

    
    # MAFFT alignment (only need to do this once)
    aligned_file = f"{virus_seq_path}_aln1.fasta"
    if not os.path.exists(aligned_file):
        mafft_aln(virus_seq_path, mafft_path)
    
    virus_seq1, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
        base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS']
    )
    
    virus_seq = {}
    virus_seq_mask = {}
    
    for seq_record in SeqIO.parse(aligned_file, "fasta"):
        seq1 = str(seq_record.seq.upper())
        seq1 = seq1.replace('*','-')
        seq1 = re.sub('[BJOUZ]', 'X', seq1)
        virus_seq[seq_record.id] = sequence_to_indexes(seq1, base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'])
        binary_mask = [1. if c == 'O' else 0. for c in N_to_O(seq1)]
        virus_seq_mask[seq_record.id] = pngs_mask_to_kemr_tensor(binary_mask, base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'])
    
    # Process each model file
    for model_path in model_files:
        # Extract fold and repeat information from filename
        filename = os.path.basename(model_path)
        fold_match = re.search(r'fold(\d+)_repeat(\d+)', filename)

        fold_idx = int(fold_match.group(1))
        repeat_idx = int(fold_match.group(2))
        
        print(f"Processing fold {fold_idx}, repeat {repeat_idx}")
        
        # Create test dataset
        test_assays = []
        j = 0
        for seq_id in virus_seq:
            test_assays.append([j, antibody, seq_id, False])
            j += 1
        
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_seq_mask)
        loader_test = t.utils.data.DataLoader(test_set, 1, shuffle=False, collate_fn=zero_padding, num_workers=0)
        
        # Load and run model
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(model_path)
        model.load_state_dict(checkpoint['model'])
        
        preds, ground_tru = eval_network_pred(model, loader_test)

        #####################################
        # Save predictions
        # output_filename = f"Predicted_threshold{threshold}_CATNAP_384_strains_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_CATNAP_oct1_all_strains_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        output_filename = f"Predicted_threshold{threshold}_CATNAP_399_more_to_add_oct1_25_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_VRC01_viruses_seqs_from_399_more_to_add_oct1_25_identity0.8_representatives_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_VRC01_01_07_Clade_model_CATNAP_399_strains_oct1_25_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_xueling_4_envs_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_Xueling_30_Env_SGA_PCR_414_m9_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_Joseph_Pacbio_LAT-03-064_RNA_demuxed_FAD_filtered_longest_orfs_aa_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        output_file = os.path.join(output_dir, output_filename)
        
        with open(output_file, "w") as f:
            f.write("Virus_ID\tPrediction\tGround_Truth\tState\n")
            for i, (seq_id, pred, truth) in enumerate(zip(virus_seq.keys(), preds, ground_tru)):
                state = 'sensitive' if pred > 0.5 else 'resistant'
                f.write(f"{seq_id}\t{pred}\t{truth}\t{state}\n")
        
        print(f"Saved predictions to {output_file}")
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def process_5folds_1_repeat(antibody, threshold, virus_seq_file, output_dir, mafft_path):
    # Find all model files for this antibody and threshold
    # model_pattern = f'model_threshold{threshold}_removed_outliers_duplicates_geomean_epitope_with_all_lineage_july16_data_{antibody}_fold*_repeat1.tar'
    model_pattern = f'model_threshold{threshold}_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_july16_data_VRC07-523-W54-LS.v3_fold*_repeat1.tar'
    model_files = glob.glob(os.path.join(FINAL_MODEL, model_pattern))
    
    print(f"Found {len(model_files)} model files for {antibody}")
    
    # Load configurations and process sequences once (common for all folds)
    base_conf = read_json_file(DEFAULT_CONF)
    # conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_epitope_with_all_lineage_removed_outliers_duplicates_geomean_{antibody}.json')
    conf_file = join(HYPERPARAM_FOLDER_ANTIBODIES, f'threshold{threshold}_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_VRC07-523-W54-LS.v3.json')
    
    conf = read_json_file(conf_file)
    conf = add_properties_from_base_config(conf, base_conf)

    #####################################
    # Process sequences once
    virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_virseqs_aa_399_more_to_add_oct1_25.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"virseqs_aa_oct1_25.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"415_m2_m6_m9_m13_aa.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"env_all_11-2025_preprocessed_CDS_aa.fasta")
    # virus_seq_path = os.path.join(virus_seq_file, f"CATNAPstrains_NON_LBUM_399_more_to_add_oct1_25.fasta")


    
    # MAFFT alignment (only need to do this once)
    aligned_file = f"{virus_seq_path}_aln1.fasta"
    if not os.path.exists(aligned_file):
        mafft_aln(virus_seq_path, mafft_path)
    
    virus_seq1, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
        base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS']
    )
    
    virus_seq = {}
    virus_seq_mask = {}
    
    for seq_record in SeqIO.parse(aligned_file, "fasta"):
        seq1 = str(seq_record.seq.upper())
        seq1 = seq1.replace('*','-')
        seq1 = re.sub('[BJOUZ]', 'X', seq1)
        virus_seq[seq_record.id] = sequence_to_indexes(seq1, base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'])
        binary_mask = [1. if c == 'O' else 0. for c in N_to_O(seq1)]
        virus_seq_mask[seq_record.id] = pngs_mask_to_kemr_tensor(binary_mask, base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'])
    
    # Process each model file
    for model_path in model_files:
        # Extract fold and repeat information from filename
        filename = os.path.basename(model_path)
        fold_match = re.search(r'fold(\d+)_repeat(\d+)', filename)

        fold_idx = int(fold_match.group(1))
        repeat_idx = int(fold_match.group(2))
        
        print(f"Processing fold {fold_idx}, repeat {repeat_idx}")
        
        # Create test dataset
        test_assays = []
        j = 0
        for seq_id in virus_seq:
            test_assays.append([j, antibody, seq_id, False])
            j += 1
        
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_seq_mask)
        loader_test = t.utils.data.DataLoader(test_set, 1, shuffle=False, collate_fn=zero_padding, num_workers=0)
        
        # Load and run model
        model = get_FC_GRU_ATT_model(conf)
        checkpoint = t.load(model_path)
        model.load_state_dict(checkpoint['model'])
        
        preds, ground_tru = eval_network_pred(model, loader_test)

        #####################################
        # Save predictions
        output_filename = f"Predicted_threshold{threshold}_CATNAP_399_more_to_add_oct1_25_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_CATNAPstrains_NON_LBUM_399_more_to_add_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_CATNAP_oct1_all_strains_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_Xueling_30_Env_SGA_PCR_415_m2_m6_m9_m13_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        # output_filename = f"Predicted_threshold{threshold}_CATNAP_69k_env_strains_fold{fold_idx}_repeat{repeat_idx}_{antibody}.txt"
        output_file = os.path.join(output_dir, output_filename)
        
        with open(output_file, "w") as f:
            f.write("Virus_ID\tPrediction\tGround_Truth\tState\n")
            for i, (seq_id, pred, truth) in enumerate(zip(virus_seq.keys(), preds, ground_tru)):
                state = 'sensitive' if pred > 0.5 else 'resistant'
                f.write(f"{seq_id}\t{pred}\t{truth}\t{state}\n")
        
        print(f"Saved predictions to {output_file}")
        
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=str, required=True, help="Threshold value for prediction")
    parser.add_argument("-a", "--antibodies", nargs='+', required=True, help="List of antibodies to process")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing input FASTA file")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("-m", "--mafft", type=str, default="mafft", help="Path to MAFFT executable")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    for antibody in args.antibodies:
        if antibody not in ANTIBODIES_LIST:
            print(f"Skipping {antibody} - not in supported antibodies list")
            continue
        
        try:
            # process_best1_of100(antibody, args.threshold, args.input_dir, args.output_dir, args.mafft)
            # process_all10folds_of_best_repeat(antibody, args.threshold, args.input_dir, args.output_dir, args.mafft)
            process_5folds_1_repeat(antibody, args.threshold, args.input_dir, args.output_dir, args.mafft)
        except Exception as e:
            print(f"Error processing {antibody}: {str(e)}")
            continue

if __name__ == '__main__':
    main()