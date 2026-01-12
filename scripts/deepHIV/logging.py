import logging
import sys
import optuna

def setup_logging():
    # optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('optuna log hypt 50'))
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('optuna log Finetune 1000 threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_final_model_5folds_1_repeat'))

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    rootLogger.addHandler(console)

    # file = logging.FileHandler('main_hypt50.log')
    file = logging.FileHandler('main_Finetune_threshold0.2_removed_outliers_duplicates_geomean_VRC01_lineage_forCV_final_model_5folds_1_repeat.log')
    file.setFormatter(formatter)
    rootLogger.addHandler(file)
