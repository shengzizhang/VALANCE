import pandas as pd
import os
import shutil
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re

# Define file paths
# input_file = "/home/yujieq/work/ML_training/bNAb-ReP/original_data/env_neu_unique_ab_removed_outliers_geomean.txt"
# output_dirs = {
#     "50": "/home/yujieq/work/ML_training/bNAb-ReP/processed_data_adding_pretrain_epitope_only/IC50_50",
#     "1": "/home/yujieq/work/ML_training/bNAb-ReP/processed_data_adding_pretrain_epitope_only/IC50_1",
#     "0.2": "/home/yujieq/work/ML_training/bNAb-ReP/processed_data_adding_pretrain_epitope_only/IC50_0.2"
# }


input_file = "/home/yujieq/work/ML_training/bNAb-ReP/original_data/env_neu_unique_ab_removed_outliers_duplicates_geomean.txt"
output_dirs = {
    "50": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_adding_pretrain_epitope_only/IC50_50",
    "1": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_adding_pretrain_epitope_only/IC50_1",
    "0.2": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_adding_pretrain_epitope_only/IC50_0.2"
}

# Create output directories if they do not exist
for key in output_dirs:
    os.makedirs(output_dirs[key], exist_ok=True)

    
training_script_path = "/home/yujieq/work/ML_training/bNAb-ReP/scripts/bNAb-ReP_GBM_training_with_pretraining.R"

# training_script_path = "/home/yujieqiao/work/ML_training/bNAb-ReP/processed_data_adding_pretrain_epitope_only/IC50_0.2/N6/bNAb-ReP_GBM_training_with_pretraining_load_params.R"

# # List of antibodies to include
# antibodies_to_include = ['561_01_18', 'N6', 'PGT121', 'VRC01',
#                          'SF12', '10-1074', 'PGT145', 'M1214_N1', '3BNC117',
#                          'PGDM1400', 'VRC38.01', 'PG9', '8ANC195', '35O22', 
#                          'VRC34.01', 'PGT135', 'b12', 'VRC-PG04', 'PGT151', 
#                          'CH01', 'DH270.6', '4E10', 'PGT128', 'HJ16', 'VRC26.25', 'VRC-CH31', '10E8', 'VRC07-523-W54-LS.v3']
antibodies_to_include = ['561_01_18', 'N6']

# Read the input file
df = pd.read_csv(input_file, sep="\t")

# Filter the dataframe to include only specified antibodies
filtered_df = df[df['Antibody'].isin(antibodies_to_include)]


# Define memory and CPU allocation for the GBM training script
num_cores = 8
memory_gb = 32

# Function to execute the GBM training script
def execute_gbm_training(filtered_df, threshold_label):
    filtered_df_abs = filtered_df.dropna(subset=['IC50'])
    
    for antibody in filtered_df_abs['Antibody'].unique():
        # Sanitize antibody name
        sanitized_antibody = antibody.replace("/", "_")
        
        # Define directory and file paths
        antibody_dir = os.path.join(output_dirs[threshold_label], sanitized_antibody)
        pretraining_filename = f"Training_{sanitized_antibody}_IC50_{threshold_label}.txt"
        pretraining_file = os.path.join(antibody_dir, pretraining_filename)

        cv_filename = f"Training_{sanitized_antibody}_IC50_{threshold_label}_antibody.txt"
        cv_file = os.path.join(antibody_dir, cv_filename)
        

        # Copy the GBM training script to the antibody directory
        shutil.copy(training_script_path, antibody_dir)
        
        # Define the command to execute the GBM training script
        # Arg4: Pretraining data: same epitope data (with all lineage members) but exclude the target antibody data
        # Arg5: Cross-validation data: antibody specific training data
        command = f"Rscript {os.path.basename(training_script_path)} {num_cores} {memory_gb} {antibody_dir} {pretraining_file} {cv_file}"
        subprocess.run(command, shell=True, cwd=antibody_dir)

        print(f"GBM training script executed for antibody {antibody} with threshold {threshold_label}.")


# Execute the GBM training script for different thresholds
# execute_gbm_training(filtered_df, "50")
# execute_gbm_training(filtered_df, "1")
execute_gbm_training(filtered_df, "0.2")