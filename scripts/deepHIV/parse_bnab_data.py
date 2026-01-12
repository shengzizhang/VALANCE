import sys
sys.path.append('/home/yujieq/work/ML_training/deep_hiv_ab_pred')
import numpy as np
import pandas as pd
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import RAWI_DATA, COMPARE_SPLITS_FOR_RAWI, ANTIBODIES_LIST
import random
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from collections import Counter

# # Split in a way that both classes have at least 1 sample (for 'N6', '561_01_18' in threhold 10 & 50, for 'VRC07-523-W54-LS.v3' in thres 0.2))
# def cross_validation_splits(cv_data, ground_truths, folds=10, repeats=10):
#     valid_splits = []
#     cv_data = np.array(cv_data)
#     ground_truths = np.array(ground_truths) 

#     # Shuffle both cv_data and ground_truths together
#     shuffled_indices = np.random.permutation(len(cv_data))
#     cv_data = cv_data[shuffled_indices]
#     ground_truths = ground_truths[shuffled_indices]
    
#     rkf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
    
#     for train_idx, test_idx in rkf.split(cv_data, ground_truths):
#         train_ground_truths = ground_truths[train_idx]
#         test_ground_truths = ground_truths[test_idx]

#         # Check if both train and test sets have at least one sample from each class
#         if len(np.unique(train_ground_truths)) > 1 and len(np.unique(test_ground_truths)) > 1:
#             valid_splits.append({
#                 'train': cv_data[train_idx].tolist(),
#                 'test': cv_data[test_idx].tolist()
#             })

#         # Stop if we have enough valid splits
#         if len(valid_splits) >= repeats * folds:
#             break
    
#     if len(valid_splits) < repeats * folds:
#         raise ValueError("Unable to create enough valid splits with at least one sample from each class.")

#     return valid_splits


# # Split in a way that both classes have at least 2 samples
# def cross_validation_splits(cv_data, ground_truths, folds=10, repeats=10):
#     valid_splits = []
#     cv_data = np.array(cv_data)
#     ground_truths = np.array(ground_truths) 

#     # Shuffle both cv_data and ground_truths together
#     shuffled_indices = np.random.permutation(len(cv_data))
#     cv_data = cv_data[shuffled_indices]
#     ground_truths = ground_truths[shuffled_indices]
    
#     rkf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
    
#     for train_idx, test_idx in rkf.split(cv_data, ground_truths):
#         train_ground_truths = ground_truths[train_idx]
#         test_ground_truths = ground_truths[test_idx]

#         # Count the number of positive and negative samples in the train and test sets
#         train_class_counts = np.bincount(train_ground_truths)
#         test_class_counts = np.bincount(test_ground_truths)

#         # Ensure there are at least 2 samples from both classes in both train and test sets
#         if len(train_class_counts) == 2 and len(test_class_counts) == 2:
#             if train_class_counts[0] >= 2 and train_class_counts[1] >= 2 and test_class_counts[0] >= 2 and test_class_counts[1] >= 2:
#                 valid_splits.append({
#                     'train': cv_data[train_idx].tolist(),
#                     'test': cv_data[test_idx].tolist()
#                 })

#         # Stop if we have enough valid splits
#         if len(valid_splits) >= repeats * folds:
#             break
    
#     if len(valid_splits) < repeats * folds:
#         raise ValueError("Unable to create enough valid splits with at least two samples from each class.")

#     return valid_splits


# # Special handling for adding synthetic random noise as additional training data(but excluded from validation)
# def cross_validation_splits(cv_data, ground_truths, folds=10, repeats=10):
#     # # Separate instances with IDs "63845" to "63894" "(for 50 random noises)
#     # fixed_train_ids = {str(i) for i in range(63845, 63895)} 
    
#     # # Separate instances with IDs "63845" to "63945" "(for 100 random noises)
#     # fixed_train_ids = {str(i) for i in range(63845, 63945)}
    
#     # # Separate instances with IDs "63845" to "64045" "(for 200 random noises)
#     # fixed_train_ids = {str(i) for i in range(63845, 64045)}
    
#     # # Separate instances with IDs "63845" to "64245" "(for 400 random noises)
#     # fixed_train_ids = {str(i) for i in range(63845, 64245)}

#     # # Separate instances with IDs "63845" to "63903" "(for 58 synthetic data from delta G > 3)
#     # fixed_train_ids = {str(i) for i in range(63845, 63903)}
    
#     fixed_train_data = [instance for instance in cv_data if str(instance) in fixed_train_ids]
#     remaining_data = [instance for instance in cv_data if str(instance) not in fixed_train_ids]
    
#     # Adjust ground_truths to match remaining_data
#     remaining_ground_truths = [ground_truths[i] for i, instance in enumerate(cv_data) if str(instance) not in fixed_train_ids]

#     # Check if lengths of remaining_data and remaining_ground_truths are consistent
#     if len(remaining_data) != len(remaining_ground_truths):
#         raise ValueError("Length mismatch between remaining_data and remaining_ground_truths")

#     # Shuffle the remaining data for cross-validation
#     random.shuffle(remaining_data)
#     remaining_data = np.array(remaining_data)
#     fixed_train_data = np.array(fixed_train_data)

#     # Generate cross-validation splits with RepeatedStratifiedKFold
#     rkf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats)
#     splits = []

#     # Create each fold
#     for train_idx, test_idx in rkf.split(remaining_data, remaining_ground_truths):
#         # Add fixed train instances to the training set and exclude from the testing set
#         train_data = np.concatenate((remaining_data[train_idx], fixed_train_data))
#         test_data = remaining_data[test_idx]

#         # Convert to lists for JSON compatibility
#         splits.append({
#             'train': train_data.tolist(),
#             'test': test_data.tolist()
#         })

#     return splits


# Regular Standrd Split (for all thresholds except for 'N6', '561_01_18' in threhold 10 & 50, for 'VRC07-523-W54-LS.v3' in thres 0.2)
def cross_validation_splits(cv_data, ground_truths, folds = 5, repeats = 1):
    random.shuffle(cv_data)
    cv_data = np.array(cv_data)
    rkf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    return [
        { 'train': cv_data[train].tolist(), 'test': cv_data[test].tolist() }
        for train, test in rkf.split(cv_data, ground_truths)
    ]

########### Added
def is_cv_splits_valid(splits, catnap):
    for split in splits:
        train, test = split['train'], split['test']
        train_ground_truths = [ int(data[3]) for data in catnap if data[0] in train ]
        test_ground_truths = [ int(data[3]) for data in catnap if data[0] in test ]
        # If there are both positive and negative outcomes return True (valid)
        if sum(train_ground_truths) == len(train_ground_truths) or sum(test_ground_truths) == len(test_ground_truths):
            return False
    return True


def create_splits_to_compare_with_rawi(catnap):
    #rawi_data = read_json_file(RAWI_DATA)
    splits = {}
    for antibody in ANTIBODIES_LIST:
        virus_ids = [data[2] for data in catnap if data[1] == antibody ]
        pretrain_data = [ data[0] for data in catnap if data[1] != antibody ]
        cv_tuples = [(antibody, virus) for virus in virus_ids]
        #Randomly sample 200, 400, 600, 800, 1000 training data to finetune the 5 antibodies which have training >1000 points (['10-1074', '3BNC117', 'PGDM1400', 'PGT121', 'VRC01'])
        # random.seed(42)
        # cv_tuples = random.sample(cv_tuples, 1000)
        cv_data = [data[0] for data in catnap if (data[1], data[2]) in cv_tuples]
        ground_truths = [data[3] for data in catnap if (data[1], data[2]) in cv_tuples]
        if len(cv_data) <1:
            continue
        elif len(cv_data) < len(cv_tuples)-20:
            print(len(cv_data))
            print(len(cv_tuples))
            print('Skipping', antibody)
            #continue
        print(antibody)
        
        
        cv_splits = cross_validation_splits(cv_data, ground_truths)

        ########### Added
        # Retry until valid splits are found
        while not is_cv_splits_valid(cv_splits, catnap):
            print('Retrying ', antibody)
            cv_splits = cross_validation_splits(cv_data, ground_truths) # Retry
            
        splits[antibody] = { 'pretraining': pretrain_data, 'cross_validation': cv_splits }
    return splits



###############################################################################################
# Added new functions below 
def calculate_ratio_for_antibody(catnap, antibody):
    """
    Calculate the positive-to-negative ratio for a specific antibody.
    """
    ground_truths = [data[3] for data in catnap if data[1] == antibody]
    total_positives = sum(1 for gt in ground_truths if gt == True)
    total_negatives = sum(1 for gt in ground_truths if gt == False)
    return total_positives / (total_positives + total_negatives)  # Positive ratio

def virusstratification(data):
    """
    Stratify data to ensure clade diversity within positive and negative samples.
    """
    letter_counts = Counter(data['clade'])
    data['split_label'] = '1'  # Default split label
    clade = ''
    for k, count in letter_counts.items():
        dfk = data[data['clade'] == k]
        if len(dfk) > 2:
            data.loc[data['clade'] == k, 'split_label'] = k
            if (len(dfk[dfk['IC50'] == True]) > 1) and (len(dfk[dfk['IC50'] == False]) > 1):
                data.loc[(data['clade'] == k) & (data['IC50'] == True), 'split_label'] = k + '2'
                data.loc[(data['clade'] == k) & (data['IC50'] == False), 'split_label'] = k + '1'
                clade = k + '1'
        elif len(dfk) == 1:
            data.loc[data['clade'] == k, 'split_label'] = clade
    return data


# def balance_samples_with_ratio(cv_tuples, catnap, num_samples, ratio, antibody):
#     """
#     Balances samples while maintaining a specific positive-to-negative ratio for an antibody.
#     """
#     # Extract ground truths for the given tuples
#     ground_truths = [(data[1], data[2], data[3]) for data in catnap if (data[1], data[2]) in cv_tuples]
#     positives = [t for t in ground_truths if t[2] == True]  # Positive outcomes
#     negatives = [t for t in ground_truths if t[2] == False]  # Negative outcomes

#     # Calculate number of positives and negatives required
#     num_positives = int(num_samples * ratio)
#     num_negatives = num_samples - num_positives

#     # Check if enough samples are available
#     if len(positives) < num_positives or len(negatives) < num_negatives:
#         print(f"Insufficient data to maintain ratio for antibody {antibody}.")
#         return None  # Indicate failure

#     # Randomly sample positives and negatives
#     random.seed(33)
#     sampled_positives = random.sample(positives, num_positives)
#     sampled_negatives = random.sample(negatives, num_negatives)

#     # Combine and shuffle
#     sampled_tuples = sampled_positives + sampled_negatives
#     random.shuffle(sampled_tuples)  # Mix positives and negatives

#     # Return only the cv_tuples
#     return [(t[0], t[1]) for t in sampled_tuples]


def balance_samples_with_clade(cv_tuples, catnap, num_samples, ratio, antibody):
    """
    Balances samples while ensuring the exact count of samples (num_samples)
    and maintaining the desired positive (IC50=True)/negative (IC50=False) ratio.
    Tries to source additional samples from unused data to preserve clade diversity.
    """
    # Step 1: Create a DataFrame from catnap (for the given antibody's virus pairs)
    ground_truths = [
        (data[1], data[2], data[3], data[4]) 
        for data in catnap 
        if (data[1], data[2]) in cv_tuples
    ]
    df = pd.DataFrame(ground_truths, columns=['antibody', 'virus', 'IC50', 'clade'])

    # Step 2: Apply stratification to label clade+1, clade+2, etc.
    stratified_data = virusstratification(df)
    grouped = stratified_data.groupby('clade')

    # Initialize
    sampled_data = []
    total_samples = 0

    # Step 3: Allocate samples proportionally to each clade
    for clade, group in grouped:
        # Proportional # of samples for this clade
        clade_ratio = len(group) / len(stratified_data)
        clade_samples = max(1, int(num_samples * clade_ratio))

        # Desired positive/negative counts for this clade
        num_positives = int(clade_samples * ratio)
        num_negatives = clade_samples - num_positives

        positives = group[group['IC50'] == True]
        negatives = group[group['IC50'] == False]

        # Handle insufficient positives
        if len(positives) < num_positives:
            num_negatives += (num_positives - len(positives))
            num_positives = len(positives)
        # Handle insufficient negatives
        if len(negatives) < num_negatives:
            num_positives += (num_negatives - len(negatives))
            num_negatives = len(negatives)

        # Sample from this clade
        sampled_positives = positives.sample(
            n=min(num_positives, len(positives)), 
            random_state=55
        ) if len(positives) > 0 else positives
        
        sampled_negatives = negatives.sample(
            n=min(num_negatives, len(negatives)), 
            random_state=55
        ) if len(negatives) > 0 else negatives

        sampled_clade_data = pd.concat([sampled_positives, sampled_negatives])
        sampled_data.append(sampled_clade_data)
        total_samples += len(sampled_clade_data)

    # Step 4: Combine, deduplicate, shuffle
    final_sampled_data = pd.concat(sampled_data).drop_duplicates()
    final_sampled_data = final_sampled_data.sample(frac=1, random_state=55)

    # Keep track of what we used
    used_indices = set(final_sampled_data.index)

    # Step 5: Check how many we have vs. how many we need
    current_count = len(final_sampled_data)
    target_count = num_samples
    needed_positives = int(target_count * ratio)
    needed_negatives = target_count - needed_positives

    # Current positives/negatives
    current_positives = final_sampled_data[final_sampled_data['IC50'] == True]
    current_negatives = final_sampled_data[final_sampled_data['IC50'] == False]
    count_pos = len(current_positives)
    count_neg = len(current_negatives)

    # --------------------------------------------------------
    #  A) If we have more samples than needed, trim the excess
    #     while trying to maintain the target ratio
    # --------------------------------------------------------
    if current_count > target_count:
        excess = current_count - target_count

        # Figure out how many positives/negatives we should keep
        # based on the target ratio
        keep_pos = min(needed_positives, count_pos)
        keep_neg = min(needed_negatives, count_neg)

        # In some edge cases, we might need to adjust
        # if keep_pos + keep_neg != target_count
        if keep_pos + keep_neg < target_count:
            # We still need more samples, pick from whichever
            # category is available
            remainder = target_count - (keep_pos + keep_neg)
            # If there's room in positives
            can_add_pos = count_pos - keep_pos
            add_pos = min(can_add_pos, remainder)
            keep_pos += add_pos
            remainder -= add_pos

            # If there's still remainder, add negatives
            can_add_neg = count_neg - keep_neg
            add_neg = min(can_add_neg, remainder)
            keep_neg += add_neg

        trimmed_positives = current_positives.sample(n=keep_pos, random_state=55)
        trimmed_negatives = current_negatives.sample(n=keep_neg, random_state=55)
        final_sampled_data = pd.concat([trimmed_positives, trimmed_negatives])
        final_sampled_data = final_sampled_data.sample(frac=1, random_state=55)

    # --------------------------------------------------------
    #  B) If we have fewer samples than needed, add from leftover
    #     while trying to maintain the target ratio
    # --------------------------------------------------------
    elif current_count < target_count:
        shortfall = target_count - current_count
        
        # Re-check how many positives/negatives we have
        current_positives = final_sampled_data[final_sampled_data['IC50'] == True]
        current_negatives = final_sampled_data[final_sampled_data['IC50'] == False]
        count_pos = len(current_positives)
        count_neg = len(current_negatives)

        # We want to end up with needed_positives and needed_negatives
        # but let's see how many we are missing:
        missing_pos = max(0, needed_positives - count_pos)
        missing_neg = max(0, needed_negatives - count_neg)

        # If ratio is flexible, we can fill whichever is missing more
        # from leftover data
        leftover = stratified_data.loc[~stratified_data.index.isin(used_indices)]
        # Shuffle leftover so we pick from diverse clades in random order
        leftover = leftover.sample(frac=1, random_state=55)

        # 1) Try to fill missing positives
        if missing_pos > 0:
            leftover_pos = leftover[leftover['IC50'] == True]
            # We can replace if leftover_pos is small
            add_pos = leftover_pos.sample(
                n=min(len(leftover_pos), missing_pos), 
                replace=True if len(leftover_pos) < missing_pos else False, 
                random_state=55
            )
            final_sampled_data = pd.concat([final_sampled_data, add_pos]).drop_duplicates()

        # 2) Try to fill missing negatives
        if missing_neg > 0:
            # Update leftover again because we may have used some positives in step 1
            used_indices = set(final_sampled_data.index)
            leftover = leftover.loc[~leftover.index.isin(used_indices)]
            leftover_neg = leftover[leftover['IC50'] == False]
            add_neg = leftover_neg.sample(
                n=min(len(leftover_neg), missing_neg),
                replace=True if len(leftover_neg) < missing_neg else False,
                random_state=55
            )
            final_sampled_data = pd.concat([final_sampled_data, add_neg]).drop_duplicates()

        # If after all that, we still don't meet target_count, fill from entire leftover 
        # ignoring ratio but still ensuring diversity
        final_count = len(final_sampled_data)
        if final_count < target_count:
            remainder_needed = target_count - final_count
            # Pull from any leftover clade
            used_indices = set(final_sampled_data.index)
            leftover = stratified_data.loc[~stratified_data.index.isin(used_indices)]
            # If leftover is too small, sample with replacement
            add_any = leftover.sample(
                n=min(len(leftover), remainder_needed),
                replace=True if len(leftover) < remainder_needed else False,
                random_state=55
            )
            final_sampled_data = pd.concat([final_sampled_data, add_any]).drop_duplicates()

    # Step 6: Final shuffle & trim/expand to exact count
    if len(final_sampled_data) > target_count:
        final_sampled_data = final_sampled_data.sample(n=target_count, random_state=55)
    elif len(final_sampled_data) < target_count:
        # Extremely unlikely if leftover had enough data, but just in case
        remainder_needed = target_count - len(final_sampled_data)
        # Sample with replacement from final_sampled_data (worst-case fallback)
        add_any = final_sampled_data.sample(n=remainder_needed, replace=True, random_state=55)
        final_sampled_data = pd.concat([final_sampled_data, add_any]).drop_duplicates()
        # If there are duplicates after sampling with replacement, re-sample to exact size
        if len(final_sampled_data) > target_count:
            final_sampled_data = final_sampled_data.sample(n=target_count, random_state=55)

    # Final shuffle to randomize order
    final_sampled_data = final_sampled_data.sample(frac=1, random_state=55)

    # Return in the needed format
    return [(row['antibody'], row['virus']) for _, row in final_sampled_data.iterrows()]


def random_sample_create_splits(catnap, num_samples=1000):
    """
    Creates splits while maintaining a constant positive-to-negative ratio for each antibody.
    """
    splits = {}
    for antibody in ANTIBODIES_LIST:
        virus_ids = [data[2] for data in catnap if data[1] == antibody]
        pretrain_data = [data[0] for data in catnap if data[1] != antibody]
        cv_tuples = [(antibody, virus) for virus in virus_ids]

        # Calculate antibody-specific positive-to-negative ratio
        positive_ratio = calculate_ratio_for_antibody(catnap, antibody)
        print(f"{antibody} positive ratio: {positive_ratio:.2f}")

        # Balance samples for the specified num_samples
        balanced_tuples = balance_samples_with_clade(cv_tuples, catnap, num_samples, positive_ratio, antibody)
        # balanced_tuples = balance_samples_with_ratio(cv_tuples, catnap, num_samples, positive_ratio, antibody)
        if not balanced_tuples:
            print(f"Not enough data to maintain ratio for {antibody} with {num_samples} samples.")
            continue

        cv_data = [data[0] for data in catnap if (data[1], data[2]) in balanced_tuples]
        ground_truths = [data[3] for data in catnap if (data[1], data[2]) in balanced_tuples]

        if len(cv_data) < 1:
            continue

        # Count positives and negatives in the sampled data
        num_positives = sum(1 for gt in ground_truths if gt == True)
        num_negatives = len(ground_truths) - num_positives
        print(f"Antibody: {antibody}, Positives: {num_positives}, Negatives: {num_negatives}")

        print(f"Processing {antibody} with {len(cv_data)} samples")

        cv_splits = cross_validation_splits(cv_data, ground_truths)
        
        # Retry until valid splits are found
        while not is_cv_splits_valid(cv_splits, catnap):
            print('Retrying ', antibody)
            cv_splits = cross_validation_splits(cv_data, ground_truths)  # Retry

        splits[antibody] = {'pretraining': pretrain_data, 'cross_validation': cv_splits}

    return splits



def exclude_epitope_lineage_create_splits(catnap, lineage_option=1, epitope_option=3):
    """
    Creates splits for cross-validation while allowing filtering based on clonal lineage and epitope.

    Parameters:
    - catnap: List of data entries.
    - lineage_option (int):
        1 -> Include all data (both lineage and non-lineage data).
        2 -> Include only non-lineage data (exclude matching lineage data).
        3 -> Include only lineage data (include entries where lineage matches or contains antibody_lineage).
    - epitope_option (int):
        1 -> Include all data (both epitope and non-epitope data).
        2 -> Include only non-epitope data (exclude matching epitope data).
        3 -> Include only epitope data (include entries where epitope matches or contains antibody_epitope).

    Returns:
    - Dictionary containing pretraining and cross-validation splits.
    """
    splits = {}

    for antibody in ANTIBODIES_LIST:
        # Get the "Clonal lineage" and "Epitope" of the current antibody
        antibody_lineage = None
        antibody_epitope = None

        for data in catnap:
            if data[1] == antibody:
                if len(data) > 5:
                    antibody_lineage = data[5]  # data[5] is the "Clonal lineage"
                if len(data) > 4:
                    antibody_epitope = data[4]  # data[4] is the "Epitope"
                break  # Stop after finding the first match

        print(f"Antibody: {antibody}, Lineage: {antibody_lineage}, Epitope: {antibody_epitope}")

        def is_epitope_match(epitope_data, reference_epitope):
            """Returns True if the two epitopes are similar (either identical or one contains the other)."""
            if epitope_data is None or reference_epitope is None:
                return False  # If either is None, they cannot match
            return epitope_data == reference_epitope or reference_epitope in epitope_data or epitope_data in reference_epitope

        def is_similar_lineage(lineage1, lineage2):
            """Returns True if the two lineages are similar (either identical or one contains the other)."""
            if lineage1 is None or lineage2 is None:
                return False  # Treat None as not similar to anything
            return lineage1 == lineage2 or lineage2 in lineage1 or lineage1 in lineage2

        # Filter pretrain_data based on lineage and epitope options
        pretrain_data = [
            data[0] for data in catnap
            if data[1] != antibody and (
                (
                    lineage_option == 1 or  # Include all lineage data
                    (lineage_option == 2 and not is_similar_lineage(data[5], antibody_lineage)) or  # Non-lineage data only
                    (lineage_option == 3 and is_similar_lineage(data[5], antibody_lineage))  # Lineage data only
                ) and
                (
                    epitope_option == 1 or  # Include all epitope data
                    (epitope_option == 2 and not is_epitope_match(data[4], antibody_epitope)) or  # Non-epitope data only
                    (epitope_option == 3 and is_epitope_match(data[4], antibody_epitope))  # Epitope data only
                )
            )
        ]

        # Get virus IDs for the current antibody
        virus_ids = [data[2] for data in catnap if data[1] == antibody]
        cv_tuples = [(antibody, virus) for virus in virus_ids]

        # Get cross-validation data and ground truths
        cv_data = [data[0] for data in catnap if (data[1], data[2]) in cv_tuples]
        ground_truths = [data[3] for data in catnap if (data[1], data[2]) in cv_tuples]

        # Skip if there's not enough data
        if len(cv_data) < 1:
            continue
        elif len(cv_data) < len(cv_tuples) - 20:
            print(len(cv_data))
            print(len(cv_tuples))
            print('Skipping', antibody)
            continue

        print(antibody)

        # Generate cross-validation splits
        cv_splits = cross_validation_splits(cv_data, ground_truths)

        # Retry until valid splits are found
        while not is_cv_splits_valid(cv_splits, catnap):
            print('Retrying ', antibody)
            cv_splits = cross_validation_splits(cv_data, ground_truths)  # Retry

        # Save the splits
        splits[antibody] = {'pretraining': pretrain_data, 'cross_validation': cv_splits}

    return splits


def VRC01_lineage_create_splits(catnap, *_, **__):
    """
    Splits policy:
      - pretraining: EXCLUDE all entries whose antibody lineage contains 'VRC01'
      - cross_validation: INCLUDE all entries whose antibody lineage contains 'VRC01'
        (i.e., CV data pools *all* VRC01-lineage antibodies, not just the current one)

    Notes:
      - lineage_option / epitope_option are ignored.
      - Returns a dict keyed by ANTIBODIES_LIST. For each key:
          * 'pretraining' is the same (non-VRC01-lineage) pool of IDs
          * 'cross_validation' splits are made from the same global VRC01-lineage pool

    For later-on fine-tuning phase:
    - CV folds are formed from a global pool of all VRC01-lineage assays, so multiple antibodies (all with VRC01-like lineage) can appear in the same fold — which is exactly why unfreezing the antibody module during CV is appropriate.
    - Pretraining data is guaranteed to be disjoint from VRC01 lineage, preventing leakage.
    
    """
    def has_vrc01_lineage(lineage):
        return isinstance(lineage, str) and ("VRC01" in lineage)
        # return isinstance(lineage, str) and (lineage == "N6")

    # --- Build global pools once ---------------------------------------------
    # Pretraining pool: all non-VRC01-lineage assay IDs
    pretrain_data_global = [
        d[0] for d in catnap
        if not has_vrc01_lineage(d[5] if len(d) > 5 else None)
    ]
    
    # CV pool: all VRC01-lineage assay IDs (+ labels for stratification)
    cv_data_global = [
        d[0] for d in catnap
        if has_vrc01_lineage(d[5] if len(d) > 5 else None)
    ]
    ground_truths_global = [
        d[3] for d in catnap
        if has_vrc01_lineage(d[5] if len(d) > 5 else None)
    ]

    # --- Print antibody distribution in each set -----------------------------
    print("\n" + "="*60)
    print("ANTIBODY DISTRIBUTION IN SPLITS")
    print("="*60)
    
    # Get unique antibodies in pretraining set
    pretrain_antibodies = set()
    for d in catnap:
        if not has_vrc01_lineage(d[5] if len(d) > 5 else None):
            pretrain_antibodies.add(d[1])  # d[1] is antibody name
    
    # Get unique antibodies in CV set
    cv_antibodies = set()
    for d in catnap:
        if has_vrc01_lineage(d[5] if len(d) > 5 else None):
            cv_antibodies.add(d[1])  # d[1] is antibody name
    
    print(f"PRETRAINING SET (non-VRC01-lineage):")
    print(f"  Total assays: {len(pretrain_data_global)}")
    print(f"  Antibodies ({len(pretrain_antibodies)}): {sorted(pretrain_antibodies)}")
    
    print(f"\nCROSS-VALIDATION SET (VRC01-lineage):")
    print(f"  Total assays: {len(cv_data_global)}")
    print(f"  Antibodies ({len(cv_antibodies)}): {sorted(cv_antibodies)}")
    
    # Print counts per antibody in each set
    print(f"\nDETAILED COUNTS:")
    print(f"  Pretraining antibodies:")
    pretrain_counts = {}
    for d in catnap:
        if not has_vrc01_lineage(d[5] if len(d) > 5 else None):
            ab = d[1]
            pretrain_counts[ab] = pretrain_counts.get(ab, 0) + 1
    for ab, count in sorted(pretrain_counts.items()):
        print(f"    {ab}: {count} assays")
    
    print(f"  CV antibodies:")
    cv_counts = {}
    for d in catnap:
        if has_vrc01_lineage(d[5] if len(d) > 5 else None):
            ab = d[1]
            cv_counts[ab] = cv_counts.get(ab, 0) + 1
    for ab, count in sorted(cv_counts.items()):
        print(f"    {ab}: {count} assays")
    
    print("="*60 + "\n")

    # --- Save antibody lists to file -----------------------------------------
    output_file = "/home/yujieq/work/ML_training/deep_hiv_ab_pred/deep_hiv_ab_pred/compare_to_Rawi_gbm/VRC01_lineage_pretrain_cv_abs_list.txt"
    
    try:
        with open(output_file, 'w') as f:
            f.write("VRC01 LINEAGE SPLITS - ANTIBODY LISTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PRETRAINING ANTIBODIES (non-VRC01-lineage)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total antibodies: {len(pretrain_antibodies)}\n")
            f.write(f"Total assays: {len(pretrain_data_global)}\n")
            f.write("Antibodies:\n")
            for ab in sorted(pretrain_antibodies):
                count = pretrain_counts.get(ab, 0)
                f.write(f"  {ab}: {count} assays\n")
            
            f.write("\n\nCROSS-VALIDATION ANTIBODIES (VRC01-lineage)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total antibodies: {len(cv_antibodies)}\n")
            f.write(f"Total assays: {len(cv_data_global)}\n")
            f.write("Antibodies:\n")
            for ab in sorted(cv_antibodies):
                count = cv_counts.get(ab, 0)
                f.write(f"  {ab}: {count} assays\n")
            
            f.write("\n\nSUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total unique antibodies in dataset: {len(pretrain_antibodies) + len(cv_antibodies)}\n")
            f.write(f"Total assays in dataset: {len(pretrain_data_global) + len(cv_data_global)}\n")
        
        print(f"✅ Antibody lists saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save antibody lists to file: {e}")

    # Quick guard: if CV pool is too small, return empty splits
    if len(cv_data_global) < 2:
        print("Not enough VRC01-lineage samples to create CV splits.")
        return {}

    # Make CV splits once from the global VRC01-lineage pool; validate & retry if needed
    cv_splits = cross_validation_splits(cv_data_global, ground_truths_global)
    while not is_cv_splits_valid(cv_splits, catnap):
        print('Retrying CV split generation on global VRC01-lineage pool')
        cv_splits = cross_validation_splits(cv_data_global, ground_truths_global)

    # --- Attach the same pools to each antibody key ---------------------------
    splits = {}
    for antibody in ANTIBODIES_LIST:
        # You can still print lineage info if useful:
        lineage = next((row[5] for row in catnap if row[1] == antibody and len(row) > 5), None)
        print(f"Antibody: {antibody}, Lineage: {lineage}")

        splits[antibody] = {
            'pretraining': pretrain_data_global,     # exclude all VRC01-lineage
            'cross_validation': cv_splits            # include only VRC01-lineage; pooled across antibodies
        }

    return splits
    

def VRC01_01_07_Clade_create_splits(catnap, *_, **__):
    """
    Splits policy:
      - pretraining: INCLUDE antibodies with epitope "GP120_CD4BS" EXCEPT specific excluded ones - These antibodies should be excluded entirely from the analysis because they are highly similar to VRC07-523LS.v34.
      - cross_validation: INCLUDE only specified VRC01-lineage antibodies
    
    Notes:
      - lineage_option / epitope_option are ignored.
      - Returns a dict keyed by ANTIBODIES_LIST. For each key:
          * 'pretraining' is the same pool of GP120_CD4BS antibodies (excluding specific ones)
          * 'cross_validation' splits are made from the specified VRC01-lineage antibodies
    """
    # Define the antibodies for cross_validation set
    cv_antibodies = [
        'VRC07-523LS.v34', 'NIH45-46', 'MinVRC01', 'VRC01-0fH/0fL',
        'VRC01-5fH/0fL', 'VRC01-5fH/10fL', 'VRC01-5fH/6fL',
        'VRC01-8fH/10fL', 'VRC01-8fH/19fL', 'VRC01-8fH/6fL',
        'VRC01.23LS', 'VRC01b', 'VRC01c', 'VRC01d', 'VRC01e',
        'VRC01f', 'VRC01g', 'VRC01h', 'VRC01i', 'VRC02', 'VRC07',
        'VRC07b', 'VRC07c', 'VRC07d', 'VRC07e', 'VRC07f'
    ]
    
    # Define antibodies to exclude from pretraining set
    excluded_pretrain_antibodies = [
        'VRC07-523-F54-LS.v3', 'VRC07-523-W54-LS.v3', 'VRC07-523LS.v11', 
        'VRC07-523LS.v14', 'VRC07-523LS.v32', 'VRC01-30fH/10fL', 
        'VRC01-30fH/19fL'
    ]

    # --- Build global pools once ---------------------------------------------
    # Pretraining pool: antibodies with epitope "GP120_CD4BS" except excluded ones
    pretrain_data_global = [
        d[0] for d in catnap
        if (d[4] == "GP120_CD4BS" if len(d) > 4 else False) and 
           (d[1] not in excluded_pretrain_antibodies if len(d) > 1 else False)
    ]

    # CV pool: only the specified VRC01-lineage antibodies
    cv_data_global = [
        d[0] for d in catnap
        if d[1] in cv_antibodies if len(d) > 1
    ]
    ground_truths_global = [
        d[3] for d in catnap
        if d[1] in cv_antibodies if len(d) > 1
    ]

    # Quick guard: if CV pool is too small, return empty splits
    if len(cv_data_global) < 2:
        print("Not enough CV antibody samples to create CV splits.")
        return {}

    # Make CV splits once from the global CV antibody pool; validate & retry if needed
    cv_splits = cross_validation_splits(cv_data_global, ground_truths_global)
    while not is_cv_splits_valid(cv_splits, catnap):
        print('Retrying CV split generation on global CV antibody pool')
        cv_splits = cross_validation_splits(cv_data_global, ground_truths_global)

    # --- Attach the same pools to each antibody key ---------------------------
    splits = {}
    for antibody in ANTIBODIES_LIST:
        # You can still print epitope info if useful:
        epitope = next((row[4] for row in catnap if row[1] == antibody and len(row) > 4), None)
        print(f"Antibody: {antibody}, Epitope: GP120_CD4BS")

        splits[antibody] = {
            'pretraining': pretrain_data_global,     # GP120_CD4BS antibodies except excluded ones
            'cross_validation': cv_splits            # specified VRC01-lineage antibodies
        }

    return splits
    

if __name__ == '__main__':
    catnap = read_json_file(CATNAP_FLAT)
    # splits = create_splits_to_compare_with_rawi(catnap)
    # splits = exclude_epitope_lineage_create_splits(catnap, lineage_option=1, epitope_option=3)
    splits = VRC01_lineage_create_splits(catnap)
    # splits = VRC01_01_07_Clade_create_splits(catnap)
    # splits = random_sample_create_splits(catnap, num_samples=1000)
    dump_json(splits, COMPARE_SPLITS_FOR_RAWI)