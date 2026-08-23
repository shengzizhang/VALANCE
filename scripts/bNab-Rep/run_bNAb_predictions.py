#!/usr/bin/env python3
"""
Script to run bNAb-ReP predictions for multiple antibodies IN PARALLEL
Each antibody gets its own prediction folder with a unique H2O port
All antibodies run simultaneously
"""

import os
import shutil
import subprocess
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time


antibodies = [
    'PGT121', 'VRC01', '10-1074', 'PGT145', 'PGDM1400',
    'VRC26.25', 'PGT151', 'PG9', '4E10', 'PGT128', 'SF12',
    '35O22', 'PGT135', 'VRC-PG04', 'CH01', 'HJ16', '10E8'
]


# Available H2O ports
h2o_ports = [
    54378, 54380, 54384, 54388, 54390, 54394, 54398, 
    54400, 54404, 54408, 54412, 54416, 54420, 54424, 54428, 
    54432, 54436, 54440, 54444, 54448, 54452, 54456, 54460
]


BASE_DIR = "/home/yujieq/work/ML_training/bNAb-ReP/removed_outliers_duplicates_no_pretraining_include_TBDs_5folds_nestedcv/IC50_50"
# TEST_FASTA_SOURCE = "/home/yujieq/work/ML_training/deep_hiv_ab_pred/final_model_predictions/CATNAP_strains/CATNAPstrains_NON_LBUM_399_more_to_add_oct1_25.fasta"
TEST_FASTA_SOURCE = "/home/yujieq/work/ML_training/BRAVE/fasta/xueling_11test_strains.fasta"
SCRIPT_SOURCE = "/home/yujieq/work/ML_training/bNAb-ReP/scripts/run_bNAb-ReP_v1.1-4.R"
MAFFT_PATH = "/home/yujieq/.conda/envs/deep_learning/bin/mafft"
# OUTPUT_PREFIX = "396_strains"
OUTPUT_PREFIX = "xueling_11test"


print_lock = threading.Lock()

def safe_print(message):
    with print_lock:
        print(message)

def modify_h2o_port(script_path, port):
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        pattern = r'h2o\.init\(ip\s*=\s*"localhost",\s*port\s*=\s*\d+\)'
        replacement = f'h2o.init(ip = "localhost", port = {port})'
        
        new_content = re.sub(pattern, replacement, content)
        
        # Also check for the other h2o.init call in bNAb.RaP.predict function
        pattern2 = r'localH2O\s*<-\s*h2o\.init\(ip\s*=\s*"localhost",\s*port\s*=\s*\d+\)'
        replacement2 = f'localH2O <- h2o.init(ip = "localhost", port = {port})'
        new_content = re.sub(pattern2, replacement2, new_content)
        
        with open(script_path, 'w') as f:
            f.write(new_content)
        
        return True
    except Exception as e:
        safe_print(f"  ERROR modifying H2O port: {e}")
        return False

def setup_prediction_folder(antibody, port):
    try:
        pred_dir = Path(BASE_DIR) / antibody / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy test FASTA file
        test_fasta_dest = pred_dir / TEST_FASTA_SOURCE.split('/')[-1]
        if not test_fasta_dest.exists():
            shutil.copy2(TEST_FASTA_SOURCE, test_fasta_dest)
        
        # Copy prediction script
        script_dest = pred_dir / "run_bNAb-ReP_v1.1-4_modified.R"
        shutil.copy2(SCRIPT_SOURCE, script_dest)
        
        if not modify_h2o_port(script_dest, port):
            return None
        
        return pred_dir
    except Exception as e:
        safe_print(f"  ERROR in setup for {antibody}: {e}")
        return None

def run_prediction(antibody, pred_dir, port):
    try:
        test_fasta = pred_dir / TEST_FASTA_SOURCE.split('/')[-1]
        reference_alignment = Path(BASE_DIR) / antibody / f"{antibody}_IC50_50_antibody_alignment_hxb2.fasta"
        model_path = Path(BASE_DIR) / antibody / "final"
        script_path = pred_dir / "run_bNAb-ReP_v1.1-4_modified.R"
        
        if not reference_alignment.exists():
            safe_print(f"  [{antibody}] ERROR: Reference alignment not found: {reference_alignment}")
            return False
        
        if not model_path.exists():
            safe_print(f"  [{antibody}] ERROR: Model not found: {model_path}")
            return False

        os.chdir(pred_dir)

        cmd = [
            "Rscript",
            str(script_path),
            antibody,
            str(test_fasta),
            MAFFT_PATH,
            OUTPUT_PREFIX,
            str(reference_alignment),
            str(model_path)
        ]
        
        safe_print(f"  [{antibody}] Running on port {port}...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 1 hour timeout
        
        log_file = pred_dir / f"{OUTPUT_PREFIX}_{antibody}_output.log"
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        
        if result.returncode == 0:
            result_file = pred_dir / f"{OUTPUT_PREFIX}_probabilities.csv"
            if result_file.exists():
                safe_print(f"  [{antibody}] SUCCESS! Results: {result_file}")
            else:
                safe_print(f"  [{antibody}] SUCCESS but result file not found?")
            return True
        else:
            safe_print(f"  [{antibody}] ERROR: Prediction failed. Check {log_file}")
            if result.stderr:
                safe_print(f"  [{antibody}] Error details: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        safe_print(f"  [{antibody}] ERROR: Prediction timed out after 1 hour")
        return False
    except Exception as e:
        safe_print(f"  [{antibody}] ERROR: {e}")
        return False

def process_antibody(antibody, port):
    safe_print(f"\n{'='*50}")
    safe_print(f"[{antibody}] Starting on port {port}")
    safe_print(f"[{antibody}] Directory: {BASE_DIR}/{antibody}/predictions")
    
    pred_dir = setup_prediction_folder(antibody, port)
    if pred_dir is None:
        safe_print(f"[{antibody}] FAILED: Could not setup prediction folder")
        return antibody, False
    
    success = run_prediction(antibody, pred_dir, port)
    
    return antibody, success

def main():
    print("="*70)
    print("bNAb-ReP Automated Prediction Pipeline (PARALLEL MODE)")
    print("="*70)
    print(f"Number of antibodies: {len(antibodies)}")
    print(f"H2O ports available: {len(h2o_ports)}")
    print(f"All antibodies will run simultaneously!")
    print("="*70)
    
    # Ensure we have enough ports
    if len(antibodies) > len(h2o_ports):
        print(f"WARNING: Not enough H2O ports! Need {len(antibodies)} but have {len(h2o_ports)}")
        print("Some antibodies will share ports (may cause conflicts)")
    
    # Assign ports to antibodies 
    antibody_port_map = {}
    for i, antibody in enumerate(antibodies):
        port = h2o_ports[i % len(h2o_ports)]
        antibody_port_map[antibody] = port
    
    print("\nPort assignments:")
    for antibody, port in antibody_port_map.items():
        print(f"  {antibody}: port {port}")
    
    print("\n" + "="*70)
    print("Starting parallel predictions...")
    print("="*70)
    
    start_time = time.time()
    
    successful = []
    failed = []

    max_workers = min(len(antibodies), 10)  # Limit to 10 parallel jobs to avoid overwhelming the system
    safe_print(f"Running with {max_workers} parallel workers (adjustable)")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_antibody = {
            executor.submit(process_antibody, antibody, antibody_port_map[antibody]): antibody
            for antibody in antibodies
        }

        for future in as_completed(future_to_antibody):
            antibody = future_to_antibody[future]
            try:
                result_antibody, success = future.result()
                if success:
                    successful.append(result_antibody)
                else:
                    failed.append(result_antibody)
            except Exception as e:
                safe_print(f"[{antibody}] Exception: {e}")
                failed.append(antibody)

    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total time: {elapsed_minutes:.2f} minutes")
    print(f"Successfully completed: {len(successful)}/{len(antibodies)}")
    
    if successful:
        print(f"\nSuccessful antibodies ({len(successful)}):")
        for antibody in successful:
            result_file = Path(BASE_DIR) / antibody / "predictions" / f"{OUTPUT_PREFIX}_probabilities.csv"
            if result_file.exists():
                print(f"  ✓ {antibody}: {result_file}")
            else:
                print(f"  ✓ {antibody}: (result file not found)")
    
    if failed:
        print(f"\nFailed antibodies ({len(failed)}):")
        for antibody in failed:
            print(f"  ✗ {antibody}")
    
    print("\n" + "="*70)
    print("LOG FILE LOCATIONS")
    print("="*70)
    for antibody in antibodies:
        log_file = Path(BASE_DIR) / antibody / "predictions" / f"{OUTPUT_PREFIX}_{antibody}_output.log"
        if log_file.exists():
            print(f"  {antibody}: {log_file}")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    exit(main())