import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse
import joblib
import esm
import torch
import numpy as np
import pandas as pd
from itertools import islice

from tabpfn.finetuning import FinetunedTabPFNClassifier

class FinetunedTabPFNClassifierFixed(FinetunedTabPFNClassifier):
    _estimator_type = "classifier"

    
parser = argparse.ArgumentParser()
parser.add_argument('--Ab', type=str, default='VRC01',help='antibody name, if combining multiple Abs for step 2 training, please seperate antibodies with comma, VRC01,VRC07. These Abs must share the same epitope.')
parser.add_argument('--model', type=str, default='',help='path to the trained model for the specific bnAb')
parser.add_argument('--reference', type=str, default='',help='path to the reference alignment file for the specific bnAb')
parser.add_argument('--file', type=str, default='',help='fasta file containing sequences for prediction')
parser.add_argument('--outfile', type=str, default='prediction.txt',help='output file name')
args = parser.parse_args()

ab=args.Ab
print(ab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval() 

def get_embeddings(data):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # Extract last layer (33 for this specific model)
    batch_tokens =batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu().detach().numpy().flatten() # Shape: [batch, seq_len, hidden_dim]
    
    return batch_labels, token_representations
def chunker(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

refids=[]
for record in SeqIO.parse(f"{args.reference}", "fasta"):
    refids.append(record.id)

# _orig_load = torch.load
# torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": torch.device("cpu")})
# final_model50 = joblib.load(f"{args.model}")
# torch.load = _orig_load

# Load the model
final_model50 = joblib.load(f"{args.model}")

standard_amino_acids = "ACDEFGHIKLMNPQRSTVWYX-"
prog=0
with open(f'{args.outfile}','w') as f:
    for i, batch in enumerate(chunker(SeqIO.parse(f'{args.file}', "fasta"), 100)):
        prog=prog+1
        with open(f'{ab}_temp.fasta', 'w') as temp:
            for record in batch:
                clean_seq = "".join([res if res in standard_amino_acids else "X" for res in record.seq.upper()])
                temp.write(f">{record.id}\n{clean_seq}\n")
        temp.close();
        # os.system(f"mafft --quiet --addfull {args.file} --thread 20 --keeplength {args.reference} > {ab}_test_aligned2Reference.fasta")
        os.system(f"mafft --quiet --addfull {ab}_temp.fasta --thread 20 --keeplength {args.reference} > {ab}_test_aligned2Reference.fasta")
        
        found_records = []
        keys=[]
        for record in SeqIO.parse(f"{ab}_test_aligned2Reference.fasta", "fasta"):
            if record.id not in refids:
                _, embedding = get_embeddings([(1, str(record.seq))])
                found_records.append(embedding)
                keys.append(record.id)
        probs = final_model50.predict_proba(np.array(found_records))
        preds = final_model50.predict(np.array(found_records))
        for k in range(len(keys)):
            f.write(f"{keys[k]}\t{probs[k]}\t{preds[k]}\n")                    
        print(f"{prog*100} finshed")

