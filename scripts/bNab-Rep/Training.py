import pandas as pd
import os
import shutil
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re

input_file = "/home/yujieq/work/ML_training/bNAb-ReP/original_data/env_neu_unique_ab_removed_outliers_duplicates_geomean_include_TBDs.txt"


output_dirs = {
    "50": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_no_pretraining_include_TBDs_5folds_nestedcv/IC50_50",
    "1": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_no_pretraining_include_TBDs_5folds_nestedcv/IC50_1",
    "0.2": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_no_pretraining_include_TBDs_5folds_nestedcv/IC50_0.2"
}

for key in output_dirs:
    os.makedirs(output_dirs[key], exist_ok=True)

    
antibodies_to_include = ['PGT121', 'VRC01', '10-1074', 'PGT145', '3BNC117', 'PGDM1400', 'VRC26.25', 'PGT151', 'PG9', '4E10', 'PGT128', 'SF12', 'N6', '35O22', 'PGT135', 'VRC-PG04', 'CH01', 'HJ16', '10E8', 'VRC34.01', 'b12', 'VRC07-523LS.v34', 'VRC03', 'VRC07', '2F5', '8ANC195']


df = pd.read_csv(input_file, sep="\t")


filtered_df = df[df['Antibody'].isin(antibodies_to_include)]

filtered_df = filtered_df[filtered_df['IC50'].notna()]


training_script_path = "/home/yujieq/work/ML_training/bNAb-ReP/scripts/bNAb-ReP_GBM_training_no_pretraining_nestedcv.R"


num_cores = 8
memory_gb = 32


def execute_gbm_training(filtered_df, threshold_label):
    filtered_df_abs = filtered_df.dropna(subset=['IC50'])
    
    for antibody in filtered_df_abs['Antibody'].unique():
        sanitized_antibody = antibody.replace("/", "_")
        
        antibody_dir = os.path.join(output_dirs[threshold_label], sanitized_antibody)
        training_filename = f"Training_{sanitized_antibody}_IC50_{threshold_label}_antibody.txt"
        training_file = os.path.join(antibody_dir, training_filename)
        
        shutil.copy(training_script_path, antibody_dir)

        command = f"Rscript {os.path.basename(training_script_path)} {num_cores} {memory_gb} {antibody_dir} {training_file}"
        subprocess.run(command, shell=True, cwd=antibody_dir)

        print(f"GBM training script executed for antibody {antibody} with threshold {threshold_label}.")

# Execute the GBM training script for different thresholds
execute_gbm_training(filtered_df, "50")
# execute_gbm_training(filtered_df, "1")
# execute_gbm_training(filtered_df, "0.2")
# execute_gbm_training(filtered_df, "2")
