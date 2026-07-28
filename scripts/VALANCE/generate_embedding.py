import glob
import os
import h5py
import numpy as np
from Bio import SeqIO
import esm
import torch
import pandas as pd
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--Ab',       type=str, default='VRC01',
                    help='antibody name')
parser.add_argument('--training', type=str,
                    default='env_neu_unique_ab_removed_outliers_duplicates_geomean_include_TBDs.txt',
                    help='path to the training data table')
parser.add_argument('--thread',   type=int, default=40,
                    help='number of threads for alignment')
parser.add_argument('--file',     type=str, default='',
                    help='fasta file containing sequences for prediction')
 
args = parser.parse_args()
print(args)
ab = args.Ab
device = 'cuda'
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()
ic50_lookup = {}
 
 
def get_embeddings(data):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu().detach().numpy().flatten()
    return batch_labels, token_representations
 
 
# get embeddings of training data
df = pd.read_csv(args.training, sep='\t')
df.dropna(subset=['IC50'], inplace=True)
df.loc[:, 'envseq'] = df['envseq'].str.replace('#', '-')
df.loc[:, 'envseq'] = df['envseq'].str.replace('*', '-')
df.loc[:, 'envseq'] = df['envseq'].str.replace('B', 'N')
df.loc[:, 'envseq'] = df['envseq'].str.replace("-", '')
df.loc[:, 'IC50']   = df['IC50'].apply(pd.to_numeric)
Ab_df = df.loc[df['Antibody'].str.match('^' + ab + '$', na=True)]
print(f"Total {ab} training dataset: {len(Ab_df)}")
ic50_lookup  = Ab_df.set_index('Virus')['IC50'].to_dict()
 
with open(f"{ab}_reference.fasta", "w") as f:
    for _, row in Ab_df.iterrows():
        f.write(f">{row['Virus']}\n{row['envseq']}\n")
 
virus_id_set = set(Ab_df['Virus'].astype(str))
 
# FIX: removed spurious extra '}' after {args.thread} in the original
os.system(f"mafft --quiet --thread {args.thread} --maxiterate 1000 --genafpair {ab}_reference.fasta > {ab}_reference_aln.fasta")
 
found_records = []
for record in SeqIO.parse(f"{ab}_reference_aln.fasta", "fasta"):
    if record.id in virus_id_set:
        found_records.append(record)
 
i = 0
with h5py.File(f"{ab}_training.h5", "w") as f:
    for record in found_records:
        raw_ic50 = ic50_lookup.get(record.id, None)
        if raw_ic50 is None:
            continue
        i += 1
        if i % 100 == 0:
            print(f"{i} finished")
 
        # FIX: pass record.id as label (was binary_label int — doesn't affect
        # embedding values, but makes batch_labels meaningful for tracing)
        _, embedding = get_embeddings([(record.id, str(record.seq))])
 
        if record.id not in f:
            dset = f.create_dataset(record.id, data=embedding, compression="gzip")
            dset.attrs['ic50'] = raw_ic50
 