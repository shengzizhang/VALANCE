from os.path import join

from deep_hiv_ab_pred.global_constants import HYPERPARAM_FOLDER

COMPARE_RAWI_FOLDER = join('deep_hiv_ab_pred', 'compare_to_Rawi_gbm')
RAWI_DATA = join(COMPARE_RAWI_FOLDER, 'Rawi_data_recereated.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold50_epitope_with_all_lineage_5folds_1_repeat.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold50_epitope_with_all_lineage_5folds_1_repeat.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold50_removed_outliers_duplicates_geomean_all_available_data.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold50_removed_outliers_geomean_epitope_with_all_lineage.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold0.2_removed_outliers_duplicates_geomean_epitope_with_all_lineage.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold50_removed_outliers_duplicates_geomean_epitope_with_all_lineage_5folds_1_repeat.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold0.2_removed_outliers_duplicates_geomean_epitope_with_all_lineage_VRC38.01_CH01.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold50_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_VRC07-523-W54-LS.v3.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold0.2_removed_outliers_duplicates_geomean_VRC01_01_07_Clade_forCV_VRC07.json')
COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_5folds_1_repeat.json')
# COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_threshold0.2_removed_outliers_duplicates_geomean_N6_lineage_forCV.json')
CATNAP_DATA = join(COMPARE_RAWI_FOLDER, 'catnap_classification.json')
MODELS_FOLDER = join(COMPARE_RAWI_FOLDER, 'models')

HYPERPARAM_FOLDER_ANTIBODIES = join(HYPERPARAM_FOLDER, 'specific_antibodies', 'ICERI_v2')
HYPERPARAM_FOLDER_ANTIBODIES_GRU = join(HYPERPARAM_FOLDER, 'specific_antibodies', 'GRU')
# Need to clone https://github.com/RedaRawi/bNAb-ReP.git
SEQ_FOLDER = '/home/yujieq/work/software/bNAb-ReP/alignments'

KMER_LEN = 'KMER_LEN'
KMER_STRIDE = 'KMER_STRIDE'

CV_FOLDS_TRIM = 10
N_TRIALS = 1000
PRUNE_TREHOLD = .05

# This is 31 because 2 antibodies were removed, they didn't have sequences
#ANTIBODIES_LIST = ['10-1074', '2F5', '2G12', '35O22', '3BNC117', '4E10', '8ANC195', 'b12', 'CH01', 'DH270.1', 'DH270.5', 'DH270.6', 'HJ16', 'NIH45-46', 'PG16', 'PG9', 'PGDM1400', 'PGT121', 'PGT128', 'PGT135', 'PGT145', 'PGT151', 'VRC-CH31', 'VRC-PG04', 'VRC01', 'VRC03', 'VRC07', 'VRC26.08', 'VRC26.25', 'VRC34.01', 'VRC38.01']
# ANTIBODIES_LIST = ['VRC01','VRC03', 'VRC07','b12','10-1074','3BNC117','PG16', 'PG9','561_01_18','M1214_N1','PGT128', 'PGT135','N6','PGT151','PGDM1400','PGT121','PGT145','SF12','2G12', 'VRC26.08', 'VRC26.25', 'VRC34.01', 'VRC38.01','35O22']

# ANTIBODIES_LIST = ['PGT121', 'VRC01', '10-1074', '3BNC117']

# ANTIBODIES_LIST = ['VRC38.01', 'PG9', '8ANC195', '35O22', 'VRC34.01', 'PGT135', 'b12', 'VRC-PG04', 'PGT151', 'CH01', 'DH270.6', '4E10', 'PGT128', 'HJ16', 'VRC26.25', 'VRC-CH31']


# ANTIBODIES_LIST = ['561_01_18', 'N6', 'PGT121', 'VRC01', 'SF12', '10-1074', 'PGT145', 'M1214_N1', '3BNC117', 'PGDM1400', 'VRC38.01', 'PG9', '8ANC195', '35O22', 'VRC34.01', 'PGT135', 'b12', 'VRC-PG04', 'PGT151', 'CH01', 'DH270.6', '4E10', 'PGT128', 'HJ16', 'VRC26.25', 'VRC-CH31'] 

# ANTIBODIES_LIST = ['VRC07-523-W54-LS.v3', '561_01_18', 'N6', 'PGT121', 'VRC01', 'SF12', '10-1074', 'PGT145', 
#            'M1214_N1', '3BNC117', 'PGDM1400', 'VRC38.01', 'PG9', '8ANC195', 
#            '35O22', 'VRC34.01', 'PGT135', 'b12', 'VRC-PG04', 'PGT151', 'CH01', 
#            'DH270.6', '4E10', 'PGT128', 'HJ16', 'VRC26.25', 'VRC-CH31', '10E8']

# ANTIBODIES_LIST = ['561_01_18', 'N6', 'PGT121', 'VRC01', 'SF12', '10-1074', 'PGT145', 
#            'M1214_N1', '3BNC117', 'PGDM1400', 'VRC38.01', 'PG9', '8ANC195', 
#            '35O22', 'VRC34.01', 'PGT135', 'b12', 'VRC-PG04', 'PGT151', 'CH01', 
#            'DH270.6', '4E10', 'PGT128', 'HJ16', 'VRC26.25', 'VRC-CH31', '10E8']

# ANTIBODIES_LIST = ['VRC38.01', 'CH01']

# ANTIBODIES_LIST = ['VRC38.01', 'PG9', '8ANC195', 
#            '35O22', 'VRC34.01', 'PGT135', 'b12', 'VRC-PG04', 'PGT151', 'CH01', 
#            'DH270.6', '4E10', 'PGT128', 'HJ16', 'VRC26.25', 'VRC-CH31', '10E8']

# ANTIBODIES_LIST = ['561_01_18', 'N6', 'PGT121']

# ANTIBODIES_LIST = ['VRC01', 'SF12', '10-1074']
                   
# ANTIBODIES_LIST = ['PGT145', 'M1214_N1']

# ANTIBODIES_LIST = ['3BNC117', 'PGDM1400']

# ANTIBODIES_LIST = ['PGDM1400', 'VRC38.01', 'PG9']

# ANTIBODIES_LIST = ['8ANC195', '35O22', 'VRC34.01']

# ANTIBODIES_LIST = ['PGT135', 'b12', 'VRC-PG04']

# ANTIBODIES_LIST = ['PGT151', 'CH01', 'DH270.6']

# ANTIBODIES_LIST = ['4E10', 'PGT128', 'HJ16']

# ANTIBODIES_LIST = ['VRC26.25', 'VRC-CH31', '10E8']

# ANTIBODIES_LIST = ['VRC01_lineage']
# ANTIBODIES_LIST = ['PGT121_lineage']
# ANTIBODIES_LIST = ['N6_lineage']
# ANTIBODIES_LIST = ['10E8_lineage']
# ANTIBODIES_LIST = ['3BNC117_lineage']
# ANTIBODIES_LIST = ['10-1074']
# ANTIBODIES_LIST = ['PGDM1400']

ANTIBODIES_LIST = ['VRC07-523-W54-LS.v3', 'VRC07-523LS.v34']

# ANTIBODIES_LIST = ['PGT121', 'VRC01', 'SF12', '10-1074', 'PGT145', 'M1214_N1', '561_01_18', 'N6', '3BNC117', 'PGDM1400']
# ANTIBODIES_LIST = ['SF12', '10-1074', 'PGT145', 'M1214_N1', '561_01_18', 'N6', '3BNC117', 'PGDM1400']




# ## Discrepant cutoff1
# ANTIBODIES_LIST = ['35O22', 'HJ16']

# ## Discrepant cutoff50
# ANTIBODIES_LIST = ['PGT151', 'DH270.6']

# # Antibodies which have training >1000 points for random sampling
# ANTIBODIES_LIST = ['10-1074', '3BNC117', 'N6', 'PGDM1400', 'PGT121', 'PGT145', 'VRC01']

# ANTIBODIES_LIST = ['10-1074','3BNC117','PGDM1400','PGT121','PGT145','VRC01']

FREEZE_ANTIBODY_AND_EMBEDDINGS = 'FREEZE_ANTIBODY_AND_EMBEDDINGS'
FREEZE_ALL = 'FREEZE_ALL'
FREEZE_ALL_BUT_LAST_LAYER = 'FREEZE_ALL_BUT_LAST_LAYER'
FREEZE_EMBEDDINGS_ONLY = 'FREEZE_EMBEDDINGS_ONLY'

