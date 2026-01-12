import pandas as pd
import os
import shutil
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re
import json

# Define file paths
input_file = "/home/yujieq/work/ML_training/bNAb-ReP/original_data/env_neu_unique_ab_removed_outliers_duplicates_geomean.txt"
output_dirs = {
    "50": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_adding_pretrain_epitope_only/IC50_50",
    "1": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_adding_pretrain_epitope_only/IC50_1",
    "0.2": "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_adding_pretrain_epitope_only/IC50_0.2"
}

# Create output directories if they do not exist
for key in output_dirs:
    os.makedirs(output_dirs[key], exist_ok=True)


r_script_path = "/home/yujieq/work/ML_training/bNAb-ReP/scripts/bNAb-ReP_preprocess_h_l_chains.R"

mafft_path = os.path.expanduser("~/.conda/envs/deep_learning/bin/mafft")

# List of antibodies to include
antibodies_to_include = ['561_01_18', 'N6', 'PGT121', 'VRC01',
                         'SF12', '10-1074', 'PGT145', 'M1214_N1', '3BNC117',
                         'PGDM1400', 'VRC38.01', 'PG9', '8ANC195', '35O22', 
                         'VRC34.01', 'PGT135', 'b12', 'VRC-PG04', 'PGT151', 
                         'CH01', 'DH270.6', '4E10', 'PGT128', 'HJ16', 'VRC26.25', 'VRC-CH31', '10E8', 'VRC07-523-W54-LS.v3']


# Print the number of antibodies in the list
print(f"Number of antibodies in the list: {len(antibodies_to_include)}")


# Read the input file
df = pd.read_csv(input_file, sep="\t")

# Filter the dataframe to include only specified antibodies
filtered_df = df[df['Antibody'].isin(antibodies_to_include)]

# Function to clean and format sequence
def clean_sequence(seq):
    # Replace any non-letter characters with dashes
    cleaned_seq = re.sub(r'[^A-Za-z]', '-', seq)
    # Replace 'B' with 'N'
    cleaned_seq = cleaned_seq.replace('B', 'N')
    # Replace '*' with '-'
    cleaned_seq = cleaned_seq.replace('*', '-')
    # Replace '#' with 'N'
    cleaned_seq = cleaned_seq.replace('#', 'N')
    return cleaned_seq

def is_epitope_match(epitope_data, reference_epitope):
    """Returns True if the two epitopes are similar (either identical or one contains the other)."""
    # Handle cases where either value is NaN/None or not a string
    if pd.isna(epitope_data) or pd.isna(reference_epitope):
        return False
    
    # Convert to strings if they aren't already
    epitope_str = str(epitope_data) if not isinstance(epitope_data, str) else epitope_data
    ref_str = str(reference_epitope) if not isinstance(reference_epitope, str) else reference_epitope
    
    return (epitope_str == ref_str or 
            ref_str in epitope_str or 
            epitope_str in ref_str)

# Load the catnap JSON data to get valid Antibody-virus pairs
with open("/home/yujieq/work/ML_training/deep_hiv_ab_pred/deep_hiv_ab_pred/catnap/catnap_flat_threshold0.2_removed_outliers_duplicates_geomean_with_epitope&lineage.json", 'r') as f:
    catnap_data = json.load(f)

# Create set of valid Antibody-virus pairs from catnap JSON
valid_pairs = set()
for item in catnap_data:
    if len(item) >= 3:  # Ensure item has at least 3 elements
        antibody = item[1]
        virus = item[2]
        valid_pairs.add((antibody, virus))
        
##################################################################################################################
##################################################################################################################
####### Include the same epitope data (with all lineage members) but exclude the target antibody data. ######################################


def process_and_save_files_epitope_only(df, filtered_df, threshold, threshold_label, mafft_path):
    for antibody, group in filtered_df.groupby("Antibody"):
        # Get the target antibody's epitope
        target_epitope = group['epitope'].iloc[0]
        
        # Skip if target epitope is NaN
        if pd.isna(target_epitope):
            print(f"Antibody: {antibody} has no epitope data, skipping...")
            continue
        
        # Find matching instances
        matched_df = df[
            df['epitope'].apply(lambda x: is_epitope_match(x, target_epitope))
        ]
        
        # Exclude the target antibody itself
        matched_df = matched_df[matched_df['Antibody'] != antibody]
        
        # Further filter to only include pairs that exist in catnap JSON
        matched_df = matched_df[
            matched_df.apply(lambda row: (row['Antibody'], row['Virus']) in valid_pairs, axis=1)
        ]
    
        # Print the number of included instances
        print(f"Antibody: {antibody}, Epitope: {target_epitope}, Matched instances: {len(matched_df)}")

        # Sanitize antibody name
        sanitized_antibody = antibody.replace("/", "_")
        
        # Create directory
        antibody_dir = os.path.join(output_dirs[threshold_label], sanitized_antibody)
        os.makedirs(antibody_dir, exist_ok=True)
        
        sequences = []
        neutralization = []

        # Store heavy/light to do concatenation in R
        # We'll write them to a small CSV: "heavy_light.csv"
        heavy_light_lines = ["id,heavy,light"]

        for _, row in matched_df.iterrows():
            antibody = str(row['Antibody']).replace(" ", "")
            epitope = str(row['epitope']).replace(" ", "")
            virus = row['Virus'].replace(" ", "")
            seq_id = f"{antibody}_{epitope}_{virus}"


            # Clean env sequence
            envseq_cleaned = clean_sequence(row['envseq'])

            # Create SeqRecord
            record = SeqRecord(Seq(envseq_cleaned), id=seq_id, description="")
            sequences.append(record)

            # Neutralization
            ic50 = float(row['IC50'])
            neut_value = 0 if ic50 < threshold else 1
            neutralization.append(str(neut_value))

            # Write heavy/light lines
            # We assume heavy/light are already "aligned" or do not need alignment
            heavy_seq = row['heavy']  # or clean_sequence(row['heavy'])
            light_seq = row['light']  # or clean_sequence(row['light'])
            heavy_light_lines.append(f"{seq_id},{heavy_seq},{light_seq}")

        # Write the final unaligned env FASTA
        # alignment_filename = f"{sanitized_antibody}_IC50_{threshold_label}_env_only.fasta"
        # neutralization_filename = f"{sanitized_antibody}_IC50_{threshold_label}_neutralization.txt"
        # Define the output file paths
        alignment_filename = f"{sanitized_antibody}_IC50_{threshold_label}_alignment.fasta"
        neutralization_filename = f"{sanitized_antibody}_IC50_{threshold_label}_neutralization.txt"
        alignment_file = os.path.join(antibody_dir, alignment_filename)
        neutralization_file = os.path.join(antibody_dir, neutralization_filename)



        with open(alignment_file, "w") as f:
            SeqIO.write(sequences, f, "fasta")

        with open(neutralization_file, "w") as f:
            f.write("\n".join(neutralization))

        # CSV for heavy/light
        heavy_light_path = os.path.join(antibody_dir, "heavy_light.csv")
        with open(heavy_light_path, "w") as f:
            f.write("\n".join(heavy_light_lines))

        # Copy R script into antibody directory
        shutil.copy(r_script_path, antibody_dir)

        # Now call the R script: 
        # Arg1: alignment FASTA (env only)
        # Arg2: neutralization file
        # Arg3: path to MAFFT
        # Arg4: heavy_light CSV
        command = f"Rscript {os.path.basename(r_script_path)} {alignment_filename} {neutralization_filename} {mafft_path} heavy_light.csv"
        # Rscript bNAb-ReP_preprocess_h_l_chains.R SF12_IC50_0.2_env_only.fasta SF12_IC50_0.2_neutralization.txt ~/.conda/envs/deep_learning/bin/mafft heavy_light.csv
        subprocess.run(command, shell=True, cwd=antibody_dir)


# # Process and save files for different thresholds
process_and_save_files_epitope_only(df, filtered_df, 0.2, "0.2", mafft_path)
# process_and_save_files_epitope_only(df, filtered_df, 1, "1", mafft_path)
# process_and_save_files_epitope_only(df, filtered_df, 50, "50", mafft_path)

##################################################################################################################
##################################################################################################################
################### Generate the antibody specific training data. ######################################

def process_and_save_files_abs_only(filtered_df, threshold, threshold_label, mafft_path):
    # First filter out rows with empty/missing IC50 values
    filtered_df_abs = filtered_df.dropna(subset=['IC50'])
    
    for antibody, group in filtered_df_abs.groupby("Antibody"):
        # Process only the data specific to this antibody
        print(f"Processing antibody: {antibody}, Instances: {len(group)}")
        
        # Sanitize antibody name for file naming
        sanitized_antibody = antibody.replace("/", "_")
        
        # Create output directory for this antibody
        antibody_dir = os.path.join(output_dirs[threshold_label], sanitized_antibody)
        os.makedirs(antibody_dir, exist_ok=True)
        
        sequences = []
        neutralization = []
        # Build CSV lines for heavy/light data; header with id, heavy, light
        heavy_light_lines = ["id,heavy,light"]

        # Loop over rows for this antibody (group contains only antibody-specific data)
        for _, row in group.iterrows():
            # Construct a unique sequence ID using antibody, epitope, and virus information.
            antibody_str = str(row['Antibody']).replace(" ", "")
            epitope = str(row['epitope']).replace(" ", "")
            virus = row['Virus'].replace(" ", "")
            seq_id = f"{antibody_str}_{epitope}_{virus}"

            # Clean env sequence
            envseq_cleaned = clean_sequence(row['envseq'])
            
            # Create a SeqRecord for the env sequence
            record = SeqRecord(Seq(envseq_cleaned), id=seq_id, description="")
            sequences.append(record)

            # Determine the neutralization value
            ic50 = float(row['IC50'])
            neut_value = 0 if ic50 < threshold else 1
            neutralization.append(str(neut_value))
            
            # Write heavy and light sequences to CSV line
            heavy_seq = row['heavy']  # assuming already formatted as desired
            light_seq = row['light']
            heavy_light_lines.append(f"{seq_id},{heavy_seq},{light_seq}")

        # Define output file names (alignment FASTA and neutralization text)
        alignment_filename = f"{sanitized_antibody}_IC50_{threshold_label}_antibody_alignment.fasta"
        neutralization_filename = f"{sanitized_antibody}_IC50_{threshold_label}_antibody_neutralization.txt"
        alignment_file = os.path.join(antibody_dir, alignment_filename)
        neutralization_file = os.path.join(antibody_dir, neutralization_filename)

        # Write the env FASTA (to be aligned in R later)
        with open(alignment_file, "w") as f:
            SeqIO.write(sequences, f, "fasta")

        # Write the neutralization values
        with open(neutralization_file, "w") as f:
            f.write("\n".join(neutralization))

        # Write the heavy/light CSV file
        heavy_light_path = os.path.join(antibody_dir, "heavy_light_antibody.csv")
        with open(heavy_light_path, "w") as f:
            f.write("\n".join(heavy_light_lines))

        # Copy the R script to the antibody directory
        shutil.copy(r_script_path, antibody_dir)

        # Call the R script with the alignment FASTA, neutralization file, mafft path, and heavy_light CSV
        command = f"Rscript {os.path.basename(r_script_path)} {alignment_filename} {neutralization_filename} {mafft_path} heavy_light_antibody.csv"
        subprocess.run(command, shell=True, cwd=antibody_dir)


# # Process and save files for different thresholds
# process_and_save_files_abs_only(filtered_df, 0.2, "0.2", mafft_path)
# process_and_save_files_abs_only(filtered_df, 1, "1", mafft_path)
# process_and_save_files_abs_only(filtered_df, 50, "50", mafft_path)