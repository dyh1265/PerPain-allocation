import pandas as pd
import functools
import operator
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.stats import ks_2samp, ttest_1samp, hedges_g
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

import os

def data_split(label='mpid_PS', seed=42):
    """
    Splits the dataset into training and testing sets, performs multiple train-test splits 
    on the training set, and saves the resulting datasets to CSV files.
    Args:
        label (str, optional): The label used to select specific columns from the dataset. 
            Defaults to 'mpid_PS'.
        seed (int, optional): The random seed used for reproducibility in train-test splits. 
            Defaults to 42.
    Returns:
        None
    Functionality:
        1. Reads the original dataset and extracts relevant columns.
        2. Removes rows with missing values in the 'end of treatment' (EOT) outcomes.
        3. Saves the 'entblindung' data to a CSV file if it does not already exist.
        4. Splits the data into training and testing sets using stratified sampling based on 
           the treatment received.
        5. Saves the training and testing datasets to CSV files.
        6. Performs 200 additional train-test splits on the training set, saving each split 
           to separate CSV files.
    """
    # read scores
    data_rct_original, outcomes_rct, outcomes_eot, t_received, entblindung = read_data()
    outcomes_eot = outcomes_eot[label + '_eot'].values
    outcomes_rct = outcomes_rct[label].values
    # remove the rows with missing values
    non_nan_val_eot = np.where(~np.isnan(outcomes_eot))
    nan_val_eot = np.where(np.isnan(outcomes_eot))

    outcomes_eot = outcomes_eot[non_nan_val_eot]
    outcomes_rct = outcomes_rct[non_nan_val_eot]
    data_rct_original = data_rct_original.loc[non_nan_val_eot]
    t_received = t_received.values[non_nan_val_eot]
    entblindung = entblindung.loc[non_nan_val_eot]

    data_rct_original.reset_index(drop=True, inplace=True)
    entblindung.reset_index(drop=True, inplace=True)

    if not os.path.exists('PERPAIN/entblindung.csv'):
        entblindung.to_csv('PERPAIN/entblindung.csv', index=False)

    print('done')

    data_rct_train, data_rct_test, t_received_train, t_received_test, outcomes_eot_train, outcomes_eot_test, \
        outcomes_rct_train, outcomes_rct_test_split = train_test_split(data_rct_original, t_received,
                                                                       outcomes_eot, outcomes_rct, test_size=0.24,
                                                                       random_state=seed, stratify=t_received)
    

    # save the test data
    data_test = pd.DataFrame(data_rct_test, columns=data_rct_test.columns)
    data_test['t'] = t_received_test
    data_test['y_baseline'] = outcomes_rct_test_split
    data_test['y_eot'] = outcomes_eot_test

    data_train = pd.DataFrame(data_rct_train, columns=data_rct_train.columns)
    data_train['t'] = t_received_train
    data_train['y_baseline'] = outcomes_rct_train
    data_train['y_eot'] = outcomes_eot_train
    if not os.path.exists(f'PERPAIN/PERPAIN_{seed}'):
        os.mkdir(f'PERPAIN/PERPAIN_{seed}')
    data_test.to_csv(f'PERPAIN/PERPAIN_{seed}/data_test_mpid.csv', index=False)
    data_train.to_csv(f'PERPAIN/PERPAIN_{seed}/data_train_mpid.csv', index=False)

    # without touching test set make 100 train test splits of the train set
    for i in range(200):
        data_rct_train_split, data_rct_test_split, t_received_train_split, t_received_test_split, outcomes_eot_train_split, outcomes_eot_test_split, \
            outcomes_rct_train_split, outcomes_rct_test_split = train_test_split(data_rct_train, t_received_train,
                                                                                 outcomes_eot_train, outcomes_rct_train,
                                                                                 test_size=0.32,
                                                                                 random_state=i)
        # save the test data
        data_test = pd.DataFrame(data_rct_test_split, columns=data_rct_test_split.columns)
        data_test['t'] = t_received_test_split
        data_test['y_baseline'] = outcomes_rct_test_split
        data_test['y_eot'] = outcomes_eot_test_split

        data_train = pd.DataFrame(data_rct_train_split, columns=data_rct_train_split.columns)
        data_train['t'] = t_received_train_split
        data_train['y_baseline'] = outcomes_rct_train_split
        data_train['y_eot'] = outcomes_eot_train_split

        data_test.to_csv(f'PERPAIN/PERPAIN_{seed}/data_test_' + str(i) + '_mpid.csv', index=False)
        data_train.to_csv(f'PERPAIN/PERPAIN_{seed}/data_train_' + str(i) + '_mpid.csv', index=False)

    # save the test data
    print('done')
    return


def prepare_data():
    """
    Prepares and processes data for analysis by reading multiple CSV files, performing data cleaning, 
    merging datasets, and saving the processed data to new CSV files.
    Returns:
        tuple: A tuple containing:
            - cols (list): A nested list of column groups used for data processing.
            - quest_names (list): A list of questionnaire names extracted from the column groups.
    Steps:
    1. Reads raw data from multiple CSV files.
    2. Removes specific rows and columns based on predefined criteria.
    3. Renames columns for consistency and merges datasets on the 'ID' column.
    4. Filters and processes data to include only relevant columns.
    5. Handles missing values by replacing placeholders and imputing missing data.
    6. Saves the processed datasets to new CSV files.
    Processed Outputs:
    - 'perpain_data/outcomes_eot.csv': Processed end-of-treatment outcomes data.
    - 'perpain_data/outcomes_rct.csv': Processed randomized controlled trial outcomes data.
    - 'perpain_data/data_rct.csv': Processed RCT data with imputed missing values.
    - 'perpain_data/entblindung_matched.csv': Processed entblindung data with therapy clustering.
    Notes:
    - The function assumes the presence of specific input files in the 'perpain_data/' directory.
    - Missing values are replaced with the median of numeric columns.
    - Therapy clustering is performed based on the 'therapie' column in the entblindung data.
    """
    # Read the is_inside from the main_perpain.py
    raw_data_rct = pd.read_csv('perpain_data/final_data_100823_rct.csv', delimiter=',', encoding='unicode_escape')
    print("raw data rct shape:", raw_data_rct.shape)
    outcomes_rct = pd.read_csv('perpain_data/final_data_outcomes_100823_rct.csv', delimiter=';',
                               encoding='unicode_escape')
    entblindung = pd.read_csv('perpain_data/Entblindung.csv', delimiter=',')
    eot_outcomes = pd.read_csv('perpain_data/eot_outcomes.csv', delimiter=';', encoding='unicode_escape')

    raw_data_control = pd.read_csv('perpain_data/final_data_100823.csv', delimiter=';', encoding='unicode_escape')
    # remove the healthy from 'Kontrolle' Arm controls as they are not part of the RCT
    raw_data_control = raw_data_control[raw_data_control['Arm'] != 'Kontrolle']

    # data_columns = raw_data_rct.columns
    # match by the ID
    outcomes_rct.rename(columns={'id': 'ID'}, inplace=True)
    eot_outcomes.columns = eot_outcomes.columns + '_eot'
    eot_outcomes.rename(columns={'id_eot': 'ID'}, inplace=True)

    # inner join to find non matching ids
    cols = [
        ['Demographie', 'alter', 'geschlecht', 'groesse', 'gewicht', 'belastungen', 'psychotherapeut', 'rauchen',
         'alkohol', 'schulabschluss', 'beruf', 'bruttoeinkommen'],
        ['EPIC', 'epic_1', 'epic_2a_summer',
         'epic_2a_winter', 'epic_2b_summer',
         'epic_2b_winter', 'epic_2c_summer', 'epic_2c_winter', 'epic_2d_summer', 'epic_2d_winter',
         'epic_2e_summer', 'epic_2e_winter', 'epic_2f_summer', 'epic_2f_winter', 'epic_3', 'epic_3_1']
        , ['BSI', 'bsi_1', 'bsi_2', 'bsi_3', 'bsi_4', 'bsi_5', 'bsi_6', 'bsi_7', 'bsi_8', 'bsi_9',
           'bsi_10', 'bsi_11', 'bsi_12', 'bsi_13', 'bsi_14', 'bsi_15', 'bsi_16', 'bsi_17', 'bsi_18', 'bsi_19',
           'bsi_20', 'bsi_21', 'bsi_22', 'bsi_23', 'bsi_24', 'bsi_25', 'bsi_26', 'bsi_27', 'bsi_28', 'bsi_29',
           'bsi_30', 'bsi_31', 'bsi_32', 'bsi_33', 'bsi_34', 'bsi_35', 'bsi_36', 'bsi_37', 'bsi_38', 'bsi_39',
           'bsi_40', 'bsi_41', 'bsi_42', 'bsi_43', 'bsi_44', 'bsi_45', 'bsi_46', 'bsi_47', 'bsi_48', 'bsi_49',
           'bsi_50', 'bsi_51', 'bsi_52', 'bsi_53'],
        ['CPG', 'cpg_1', 'cpg_2', 'cpg_3', 'cpg_4', 'cpg_5', 'cpg_6',
         'cpg_7'], ['CQR5', 'cqr5_1', 'cqr5_2', 'cqr5_3', 'cqr5_4', 'cqr5_5'],
        ['EQ5D5L', 'eq5d5l_1', 'eq5d5l_2',
         'eq5d5l_3', 'eq5d5l_4', 'eq5d5l_5', 'eq5d5l_scale'],
        ['MPID', 'mpid_1_1', 'mpid_1_2', 'mpid_1_3', 'mpid_1_4',
         'mpid_1_5', 'mpid_1_6', 'mpid_1_7', 'mpid_1_8', 'mpid_1_9', 'mpid_1_10', 'mpid_1_11', 'mpid_1_12',
         'mpid_1_13', 'mpid_1_14', 'mpid_1_15', 'mpid_1_16', 'mpid_1_17', 'mpid_1_18', 'mpid_1_19', 'mpid_1_20',
         'mpid_1_21', 'mpid_1_22', 'mpid_2_1', 'mpid_2_2', 'mpid_2_3', 'mpid_2_4', 'mpid_2_5', 'mpid_2_6',
         'mpid_2_7', 'mpid_2_8', 'mpid_2_9', 'mpid_2_10', 'mpid_2_11', 'mpid_3_01', 'mpid_3_02', 'mpid_3_03',
         'mpid_3_04', 'mpid_3_05', 'mpid_3_06', 'mpid_3_07', 'mpid_3_08', 'mpid_3_09', 'mpid_3_10', 'mpid_3_11',
         'mpid_3_12', 'mpid_3_13', 'mpid_3_14', 'mpid_3_15', 'mpid_3_16', 'mpid_3_17', 'mpid_3_18'],
        ['ODI', 'odi_1',
         'odi_2', 'odi_3', 'odi_4', 'odi_5', 'odi_6', 'odi_7', 'odi_8', 'odi_9', 'odi_10'],
        ['SES', 'ses_01_quaelend',
         'ses_02_grausam', 'ses_03_erschoepfend', 'ses_04_heftig', 'ses_05_moerderisch', 'ses_06_elend',
         'ses_07_schauderhaft', 'ses_08_scheusslich', 'ses_09_schwer', 'ses_10_entnervend', 'ses_11_marternd',
         'ses_12_furchtbar', 'ses_13_unertraeglich', 'ses_14_laehmend', 'ses_15_schneidend', 'ses_16_klopfend',
         'ses_17_brennend', 'ses_18_reissend', 'ses_19_pochend', 'ses_20_gluehend', 'ses_21_stechend',
         'ses_22_haemmernd', 'ses_23_heiss', 'ses_24_durchstossend', 'ses_25_andere'],
        ['SSD12', 'ssd12_1', 'ssd12_2', 'ssd12_3', 'ssd12_4', 'ssd12_5', 'ssd12_6', 'ssd12_7', 'ssd12_8',
         'ssd12_9', 'ssd12_10', 'ssd12_11', 'ssd12_12'], ['SSS8', 'sss8_1', 'sss8_2', 'sss8_3', 'sss8_4',
                                                          'sss8_5', 'sss8_6', 'sss8_7', 'sss8_8'],
        ['TICS', 'tics_1', 'tics_2', 'tics_3', 'tics_4', 'tics_5',
         'tics_6', 'tics_7', 'tics_8', 'tics_9', 'tics_10', 'tics_11', 'tics_12', 'tics_13', 'tics_14',
         'tics_15', 'tics_16', 'tics_17', 'tics_18', 'tics_19', 'tics_20', 'tics_21', 'tics_22', 'tics_23',
         'tics_24', 'tics_25', 'tics_26', 'tics_27', 'tics_28', 'tics_29', 'tics_30', 'tics_31', 'tics_32',
         'tics_33', 'tics_34', 'tics_35', 'tics_36', 'tics_37', 'tics_38', 'tics_39', 'tics_40', 'tics_41',
         'tics_42', 'tics_43', 'tics_44', 'tics_45', 'tics_46', 'tics_47', 'tics_48', 'tics_49', 'tics_50',
         'tics_51', 'tics_52', 'tics_53', 'tics_54', 'tics_55', 'tics_56', 'tics_57'], ['WI7', 'wi7_1',
                                                                                        'wi7_2', 'wi7_3',
                                                                                        'wi7_4', 'wi7_5',
                                                                                        'wi7_6', 'wi7_7'],
        ['WOMAC', 'womac_1_1', 'womac_1_2',
         'womac_1_3', 'womac_1_4', 'womac_1_5', 'womac_2_1', 'womac_2_2', 'womac_3_1', 'womac_3_2',
         'womac_3_3', 'womac_3_4', 'womac_3_5', 'womac_3_6', 'womac_3_7', 'womac_3_8', 'womac_3_9',
         'womac_3_10', 'womac_3_11', 'womac_3_12', 'womac_3_13', 'womac_3_14'], ['WPI', 'wpi_1', 'wpi_2',
                                                                                 'wpi_3', 'wpi_4', 'wpi_5',
                                                                                 'wpi_6', 'wpi_7', 'wpi_8',
                                                                                 'wpi_9', 'wpi_10', 'wpi_11',
                                                                                 'wpi_12',
                                                                                 'wpi_13', 'wpi_14', 'wpi_15',
                                                                                 'wpi_16', 'wpi_17', 'wpi_18',
                                                                                 'wpi_19', 'wpi_20',
                                                                                 'wpi_22'],
        ['PANAS', 'panas_01', 'panas_02', 'panas_03', 'panas_04', 'panas_05',
         'panas_06', 'panas_07', 'panas_08', 'panas_09', 'panas_10', 'panas_11', 'panas_12', 'panas_13',
         'panas_14', 'panas_15', 'panas_16', 'panas_17', 'panas_18', 'panas_19', 'panas_20'],
        ['HADS', 'hads_1', 'hads_2', 'hads_3', 'hads_4', 'hads_5', 'hads_6', 'hads_7',
         'hads_8', 'hads_9', 'hads_10', 'hads_11', 'hads_12', 'hads_13', 'hads_14'], ['PSQI',
                                                                                      'psqi_2',
                                                                                      'psqi_4',
                                                                                      'psqi_5',
                                                                                      'psqi_6',
                                                                                      'psqi_7',
                                                                                      'psqi_8',
                                                                                      'psqi_9',
                                                                                      'psqi_10',
                                                                                      'psqi_11',
                                                                                      'psqi_12',
                                                                                      'psqi_13',
                                                                                      'psqi_14',
                                                                                      'psqi_16',
                                                                                      'psqi_17',
                                                                                      'psqi_18',
                                                                                      'psqi_19',
                                                                                      'psqi_20'],
        ['CTQ', 'ctq_01', 'ctq_02', 'ctq_03', 'ctq_04', 'ctq_05',
         'ctq_06', 'ctq_07', 'ctq_08', 'ctq_09', 'ctq_10', 'ctq_11', 'ctq_12', 'ctq_13', 'ctq_14',
         'ctq_15', 'ctq_16', 'ctq_17', 'ctq_18', 'ctq_19', 'ctq_20', 'ctq_21', 'ctq_22', 'ctq_23',
         'ctq_24', 'ctq_25', 'ctq_26', 'ctq_27', 'ctq_28'],
        ['FSS', 'fss_1', 'fss_2', 'fss_3', 'fss_4', 'fss_5', 'fss_6', 'fss_7', 'fss_8'],
        ['PSS', 'pss_1', 'pss_2', 'pss_3', 'pss_4', 'pss_5', 'pss_6', 'pss_7', 'pss_8', 'pss_9', 'pss_10'],
        ['RS11', 'rs_1', 'rs_2', 'rs_3', 'rs_4', 'rs_5', 'rs_6', 'rs_7', 'rs_8', 'rs_9', 'rs_10', 'rs_11'],
        ['SF12', 'sf12_1', 'sf12_2', 'sf12_3', 'sf12_4', 'sf12_5', 'sf12_6', 'sf12_7', 'sf12_8', 'sf12_9',
         'sf12_10', 'sf12_11', 'sf12_12'],
        ['FABQ', 'fabq_1', 'fabq_2', 'fabq_3', 'fabq_4', 'fabq_5', 'fabq_6', 'fabq_7', 'fabq_8',
         'fabq_9', 'fabq_10', 'fabq_11', 'fabq_12', 'fabq_13', 'fabq_14', 'fabq_15', 'fabq_16'],
        ['FPQ', 'fpq_1', 'fpq_2', 'fpq_3', 'fpq_4', 'fpq_5', 'fpq_6', 'fpq_7', 'fpq_8', 'fpq_9',
         'fpq_10', 'fpq_11', 'fpq_12', 'fpq_13', 'fpq_14', 'fpq_15', 'fpq_16', 'fpq_17', 'fpq_18',
         'fpq_19', 'fpq_20', 'fpq_21', 'fpq_22', 'fpq_23', 'fpq_24', 'fpq_25', 'fpq_26', 'fpq_27',
         'fpq_28', 'fpq_29', 'fpq_30'], ['PRSS', 'prss_1', 'prss_2', 'prss_3', 'prss_4', 'prss_5',
                                         'prss_6', 'prss_7', 'prss_8', 'prss_9', 'prss_10', 'prss_11',
                                         'prss_12', 'prss_13',
                                         'prss_14', 'prss_15', 'prss_16', 'prss_17', 'prss_18'],
        ['PDS', "pds_1_1", "pds_1_2", "pds_1_3", "pds_1_4", "pds_1_5", "pds_1_6", "pds_1_7", "pds_1_8",
         "pds_1_9", "pds_1_10", "pds_1_11", "pds_1_12"],
    ]
    # flatten the list of lists
    flat_cols = functools.reduce(operator.iconcat, cols, [])
    # questionnaire names
    quest_names = [name[0] for name in cols]
    # other columns
    other_cols = [col for col in flat_cols if col not in quest_names]
    other_cols.append('ID')
    # remove the columns 
    variables_to_remove = ['wpi_21', 'wpi_23', 'PDS', 'pds_1_1', 'pds_1_2', 'pds_1_3',
                           'pds_1_4', 'pds_1_5', 'pds_1_6', 'pds_1_7', 'pds_1_8', 'pds_1_9', 'pds_1_10',
                           'pds_1_11', 'pds_1_12', 'pds_1_12_txt', 'pds_1_belast_ereig', 'pds_jahr',
                           'pds_2_1', 'pds_2_2', 'pds_2_3', 'pds_2_4', 'pds_2_5', 'pds_2_6', 'pds_2_7',
                           'pds_3_1', 'pds_3_2', 'pds_3_3', 'pds_3_4', 'pds_3_5', 'pds_3_6', 'pds_3_7',
                           'pds_3_8', 'pds_3_9', 'pds_3_10', 'pds_3_11', 'pds_3_12', 'pds_3_13',
                           'pds_3_14', 'pds_3_15', 'pds_3_16', 'pds_3_17', 'pds_3_seit',
                           'pds_3_beginn', 'pds_4_1', 'pds_4_2', 'pds_4_3', 'pds_4_4',
                           'pds_4_5', 'pds_4_6', 'pds_4_7', 'pds_4_8', 'pds_4_9', 'KLL', 'kll_1', 'kll_1_1',
                           'kll_1_1_1',
                           'kll_2', 'kll_2_1', 'kll_2_1_1', 'kll_3', 'kll_3_1', 'kll_3_1_1', 'kll_4', 'kll_4_1',
                           'kll_4_1_1', 'kll_5', 'kll_5_1', 'kll_5_1_1', 'kll_6', 'kll_6_1', 'kll_6_1_1', 'kll_7',
                           'kll_7_1', 'kll_7_1_1', 'kll_8', 'kll_8_1', 'kll_8_1_1', 'kll_9', 'kll_9_1', 'kll_9_1_1',
                           'kll_10', 'kll_10_1', 'kll_10_1_1', 'kll_11', 'kll_11_1', 'kll_11_1_1', 'kll_12',
                           'kll_12_1', 'kll_12_1_1', 'kll_13', 'kll_13_1', 'kll_13_1_1', 'kll_14', 'kll_14_1',
                           'kll_14_1_1', 'kll_15', 'kll_15_1', 'kll_15_1_1', 'kll_16', 'kll_16_1', 'kll_16_1_1',
                           'kll_17', 'kll_17_1', 'kll_17_1_1', 'kll_18', 'kll_18_1', 'kll_18_1_1', 'kll_19',
                           'kll_19_1', 'kll_19_1_1', 'kll_20', 'kll_20_1', 'kll_20_1_1', 'kll_21', 'kll_21_1',
                           'kll_21_1_1', 'kll_22', 'kll_22_1', 'kll_22_1_1', 'kll_23', 'kll_23_1', 'kll_23_1_1',
                           'kll_24', 'kll_24_1', 'kll_24_1_1', 'kll_25', 'kll_25_1', 'kll_25_1_1', 'kll_26',
                           'kll_26_1', 'kll_26_1_1', 'kll_27', 'kll_27_1', 'kll_27_1_1', 'kll_28', 'kll_28_1',
                           'kll_28_1_1', 'kll_29', 'kll_29_1', 'kll_29_1_1', 'kll_30', 'kll_30_1', 'kll_30_1_1',
                           'kll_31', 'kll_31_1', 'kll_31_1_1', 'kll_32', 'psqi_15', 'psqi_14_1', 'psqi_3', 'psqi_1',
                           'ses_25_andere1', 'epic_4', 'facharzt', 'facharzt2',
                           'facharzt_kontakte', 'hausarzt', 'hausarzt_kontakte', 'psychiater',
                           'psychiater_kontakte', 'psychotherapeut', 'psychotherapeut_kontakte',
                           'belastungen_welche', 'rauchen', 'zigarettenanzahl', 'alkohol', 'alkoholmenge1',
                           'alkoholmenge2',
                           'alkoholmenge3', 'schwanger', 'alter', 'staatsangehoerigkeit', 'familienstand', 'kinder',
                           'kinderanzahl', 'kindesalter', 'lebensgemeinschaft', 'muttersprache',
                           'and_muttersprache', 'and_staatsangehoerigkeit',
                           'herkunftsland', 'and_herkunftsland', 'herkunftsland_mutter', 'and_herkunftsland_mutter',
                           'herkunftsland_vater',
                           'and_herkunftsland_vater', 'schulabschluss', 'bildungsjahre', 'beruf', 'bruttoeinkommen',
                           'krankschreiben_1',
                           'wohnsituation', 'Einschluss', 'Arm', 'Diagnose_validiert', 'Hauptdiagnose',
                           'belastungen_welche',
                           'hauptdiagnosen',
                           'DiagnoseprimärerSchmerzerkrankungen', 'DiagnosesekundäreSchmerzerkrankungen',
                           'PREV', 'weiterekrankheiten', 've_ns', 'pgic_ns', 've_atemweg',
                           'pgic_atemweg', 've_cardio', 'pgic_cardio', 've_magendarm', 'pgic_magendarm', 've_leber',
                           'pgic_leber', 've_niere', 'pgic_niere', 've_stoffwechsel', 'pgic_stoffwechsel',
                           've_haut',
                           'pgic_haut', 've_muskelskelett', 'pgic_muskelskelett', 've_seelische_leiden',
                           'pgic_seelische_leiden',
                           'andere_erkrankungen', 'unvertraeglichkeiten', 'unvertraeglichkeiten_2', 'vb_pt',
                           'vb_pt_art___1',
                           'vb_pt_art___2', 'vb_pt_art___3', 'vb_pt_art___4', 'vb_pt_art___5', 'vb_pt_art___6',
                           'vb_pt_art___7',
                           'vb_pt_art___8', 'vp_pt_art_andere', 'vb_pt_aktuell', 'vb_pt_abbruch', 'vb_op',
                           'vb_op_art', 'pgic_op',
                           'vb_infusion', 'pgic_infusion', 'vb_anzahl_infusion', 'vb_einspritzungen',
                           'pgic_einspritzungen',
                           'vb_anzahl_einspritzungen', 'vb_triggerpunkt', 'pgic_triggerpunkt',
                           'vb_anzahl_triggerpunkt',
                           'vb_einspritzung_rueckenmark', 'pgic_einspritzung_rueckenmark',
                           'vb_anzahl_einspritzung_rueckenmark',
                           'vb_scs', 'pgic_scs', 'vb_kg', 'vb_kg1', 'vb_kg2', 'pgic_kg_1', 'pgic_kg_2', 'pgic_kg_3',
                           'vb_anzahl_kg', 'vb_massage', 'pgic_massage', 'vb_anzahl_massage', 'vb_sport',
                           'pgic_sport',
                           'vb_sportart___1', 'vb_sportart___2', 'vb_sportart___3', 'vb_sportart___4',
                           'vb_sportart___5',
                           'vb_sportart___6', 'vb_sportart___7', 'vb_sportart___8', 'vb_sportart___9',
                           'vb_sportart___10',
                           'vb_sportart___11', 'vb_sportart___12', 'vb_sportart___13', 'vb_sportart___14',
                           'vb_sonstigetherapien___1', 'vb_sonstigetherapien___2', 'vb_sonstigetherapien___3',
                           'vb_sonstigetherapien___4', 'vb_sonstigetherapien___5', 'vb_sonstigetherapien___6',
                           'vb_sonstigetherapien___7', 'vb_sonstigetherapien___8', 'vb_sonstigetherapien___9',
                           'vb_sonstigetherapien___10', 'vb_sonstigetherapien___11', 'vb_sonstigetherapien___12',
                           'vb_sonstigetherapien___13', 'vb_sonstigetherapien___14', 'vb_sonstigetherapien___15',
                           'vb_sonstigetherapien___16', 'vb_sonstigetherapien___17', 'vb_sonstigetherapien___18',
                           'vb_sonstigetherapien___19', 'vb_sonstigetherapien___20',
                           'vb_sonstigetherapien_sonstige', 'CORONA', 'corona_1', 'corona_2',
                           'corona_4', 'corona_5', 'corona_6', 'corona_7', 'corona_8', 'corona_9', 'corona_10',
                           'corona_11', 'corona_12', 'corona_13', 'corona_14', 'corona_15', 'corona_16',
                           'corona_17',
                           'corona_18', 'corona_19', 'corona_20', 'corona_21', 'corona_22', 'corona_23',
                           'corona_24',
                           'corona_25', 'corona_26', 'corona_27', 'corona_28', 'corona_29', 'corona_30',
                           'corona_31',
                           'corona_32', 'corona_33', 'corona_34', 'corona_35', 'corona_36', 'corona_37',
                           'corona_38',
                           'corona_39', 'corona_40', 'corona_41', 'corona_42', 'corona_43', 'corona_44',
                           'corona_45',
                           'corona_46___1', 'corona_46___2', 'corona_47___1', 'corona_47___2', 'corona_47___3',
                           'corona_48___1', 'corona_48___2', 'corona_48___3', 'corona_48___4', 'corona_49',
                           'corona_50', 'VAR00001']
    # take the selected data
    data_rct = raw_data_rct[other_cols]
    data_baseline = raw_data_control[other_cols]
    # check which ids have the eot outcomes and data rct
    eot_outcomes_data = pd.merge(eot_outcomes, data_rct, on='ID', how='inner')
    
    # merge with eot outcomes_rct
    eot_outcomes_data = pd.merge(eot_outcomes, data_rct, on='ID', how='inner')
    # merge with entblindung
    entblindung_eot_outcomes_data = pd.merge(entblindung, eot_outcomes_data, on='ID', how='inner')
    # merge with outcomes_rct
    outcomes_entblindung_eot_outcomes_data = pd.merge(outcomes_rct, entblindung_eot_outcomes_data, on='ID', how='inner')

    data_rct = outcomes_entblindung_eot_outcomes_data[data_rct.columns]
    entblindung_mathed = outcomes_entblindung_eot_outcomes_data[entblindung.columns]

    t_by_clustering = []
    therapie = entblindung_mathed['therapie']
    for s in therapie:
        if s == 'PERT':
            t_by_clustering.append(2)
        elif s == 'EMI':
            t_by_clustering.append(1)
        else:
            t_by_clustering.append(0)
    entblindung_mathed['t_by_clustering'] = t_by_clustering
    eot_outcomes = outcomes_entblindung_eot_outcomes_data[eot_outcomes.columns]
    outcomes_rct = outcomes_entblindung_eot_outcomes_data[outcomes_rct.columns]

    pds_vars = ["pds_1_1", "pds_1_2", "pds_1_3", "pds_1_4", "pds_1_5", "pds_1_6", "pds_1_7", "pds_1_8",
                "pds_1_9", "pds_1_10", "pds_1_11", "pds_1_12"]
    data_rct[pds_vars] = data_rct[pds_vars].replace({np.nan, 0})

    # impute the missing values
    data_rct[data_rct == -99] = np.nan

    for col in data_rct.columns:
        data_rct[col] = pd.to_numeric(data_rct[col], errors='ignore')
    
    ids = data_rct['ID']
    data_rct = data_rct.applymap(lambda x: x if str(x).replace('.', '', 1).isdigit() else np.nan)
    data_rct = data_rct.fillna(data_rct.median(numeric_only=True))
    data_rct['ID'] = ids

    # save the data
    eot_outcomes.to_csv('perpain_data/outcomes_eot.csv', index=False)
    outcomes_rct.to_csv('perpain_data/outcomes_rct.csv', index=False)
    data_rct.to_csv('perpain_data/data_rct.csv', index=False)
    entblindung_mathed.to_csv('perpain_data/entblindung_matched.csv', index=False)

    return cols, quest_names


def read_data():
    """
    Reads and processes multiple CSV files containing data for analysis.

    Returns:
        tuple: A tuple containing the following:
            - data_rct (pd.DataFrame): Data from 'perpain_data/data_rct.csv'.
            - outcomes_rct (pd.DataFrame): Data from 'perpain_data/outcomes_rct.csv'.
            - outcomes_eot (pd.DataFrame): Data from 'perpain_data/outcomes_eot.csv'.
            - t_received (pd.Series): The 't_recived' column from 'perpain_data/entblindung_matched.csv'.
            - entblindung (pd.DataFrame): Data from 'perpain_data/entblindung_matched.csv'.
    """
    data_rct = pd.read_csv('perpain_data/data_rct.csv')
    #find text in columns and replace them with none
    outcomes_rct = pd.read_csv('perpain_data/outcomes_rct.csv')
    outcomes_eot = pd.read_csv('perpain_data/outcomes_eot.csv')
    entblindung = pd.read_csv('perpain_data/entblindung_matched.csv')
    t_received = entblindung['t_recived']
    # get the validation dataset
    return data_rct, outcomes_rct, outcomes_eot, t_received, entblindung


def select_variables(label, outcomes_to_quest):
    """
    Selects and processes variables for analysis based on the given label and outcomes.

    This function performs the following steps:
    1. Reads and prepares data from various sources.
    2. Identifies and excludes rows with missing values in the end-of-treatment (EOT) outcomes.
    3. Combines demographic data with baseline scores and imputes missing values.
    4. Removes highly correlated features (correlation > 0.7) except for the target label.
    5. Scales the data using MinMaxScaler.
    6. Computes Pearson correlation coefficients between features and the target outcome for different treatments.
    7. Aggregates correlation scores across multiple training splits to compute a final score for each feature.

    Args:
        label (str): The target variable label to analyze.
        outcomes_to_quest (dict): A mapping of outcome variables to their corresponding questionnaire names.

    Returns:
        None: The function prints the top 4 features with the highest aggregated scores.

    Notes:
        - The function assumes the existence of specific files and data structures, such as 
          "PERPAIN/data_train_{i}_mpid.csv" for training splits and predefined demographic columns.
        - The correlation threshold for feature selection is set to 0.7.
        - The final scores are averaged over 200 training splits.
    """
    # select the variables
    data_rct, scores_baseline_rct, outcomes_eot_rct, t_received, _ = read_data()
    eot = outcomes_eot_rct[label+'_eot'] 
    # find indices non NaN in eot
    nan_indices = eot[eot.isna()].index
    # find their ids
    ids_nan = outcomes_eot_rct.loc[nan_indices, 'ID']
    outcomes_eot_rct[list(ids_nan)]
    covar_sum = pd.DataFrame()
    for i in range(200):
        train_split = pd.read_csv(f"PERPAIN/data_train_{i}_mpid.csv")
        #select scores by same id patients as in the train set
        scores_baseline_rct_copy = scores_baseline_rct[scores_baseline_rct['ID'].isin(train_split['ID'])].copy()
        # # 'panas'
        demographie = ['alter', 'geschlecht', 'groesse', 'gewicht', 'belastungen', 'epic_1', 'epic_2a_summer',
                    'epic_2a_winter', 'epic_2b_summer',
                    'epic_2b_winter', 'epic_2c_summer', 'epic_2c_winter', 'epic_2d_summer', 'epic_2d_winter',
                    'epic_2e_summer', 'epic_2e_winter', 'epic_2f_summer', 'epic_2f_winter', 'epic_3', 'epic_3_1']
        data_demographie = data_rct[demographie]
        # combine demographie and outcomes

        scores_baseline_rct_copy = pd.concat([data_demographie, scores_baseline_rct_copy], axis=1)
        scores_baseline_rct_copy = scores_baseline_rct_copy[
        scores_baseline_rct_copy['ID'].isin(train_split['ID'])
            ]
        # drop the ID
        scores_baseline_rct_copy = scores_baseline_rct_copy.drop('ID', axis=1).copy()
        outcomes_eot_rct_copy = outcomes_eot_rct[outcomes_eot_rct['ID'].isin(train_split['ID'])].copy()
        # impute the outcomes baseline
        data_baseline_rct = scores_baseline_rct_copy.fillna(scores_baseline_rct_copy.median())

        # Create correlation matrix
        corr_matrix = data_baseline_rct.corr().abs()
        #Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        #Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.7) and not column == label]
        #Drop features
        data_baseline_rct.drop(to_drop, axis=1, inplace=True)
        outcomes_labels_eot = label + '_eot'
        y_eot = outcomes_eot_rct_copy[outcomes_labels_eot]
        # remove the rows with missing values
        na_val_eot = y_eot.isna()
        y_eot = y_eot.loc[~na_val_eot]
        y_baseline_rct = data_baseline_rct[label]
        y_baseline_rct = y_baseline_rct.loc[~na_val_eot]
        data_baseline_rct = data_baseline_rct.loc[~na_val_eot]
        # scale the data
        data_scaler_rct = MinMaxScaler().fit(data_baseline_rct)
        colnames = data_baseline_rct.columns
        data_rct_transformed = data_scaler_rct.transform(data_baseline_rct)
        label_name = label + '_eot'
        y_scaler = MinMaxScaler().fit(y_eot.values.reshape(-1, 1))
        y_eot_scaled = y_scaler.transform(y_eot.values.reshape(-1, 1))
        # print(
        #     '##################################SELECTION BY PEARSON CORRELATION###############################################')
        labeled_set = pd.DataFrame(data_rct_transformed.copy(), columns=colnames)
        # add outcome
        labeled_set[label_name] = y_eot_scaled

        treatments = [0, 1, 2]
        covar_set = pd.DataFrame()
        for t in treatments:
            labeled_t = labeled_set[t_received == t]
            cor = labeled_t.corr()
            #cor_target = abs(cor[label_name]).sort_values(ascending=False)
            # sort cor_target
            cor_target = abs(cor[label_name])
            if t == 0:
                covar_set['EDTT'] = cor_target
            elif t == 1:
                covar_set['EMDI'] = cor_target
            else:
                covar_set['PERT'] = cor_target
        covar = covar_set.round(2)
        covar[f"SUM_{i}"] = covar.sum(axis=1)
        covar_sum = pd.concat([covar_sum, covar[f"SUM_{i}"]], axis=1)

    # compute the final score by summing the SUM_i columns
    covar_sum['SUM'] = covar_sum[[col for col in covar_sum.columns if col.startswith('SUM_')]].sum(axis=1)
    covar_sum['SUM'] = covar_sum['SUM'] / 200.0
    covar_sum = covar_sum.sort_values(by=[f"SUM"], ascending=False)

    print(covar_sum["SUM"].head(4))


def get_data_stats():
    from utils.stats import cohens_d, hedges_g, compute_confidence_interval
    """
    Analyzes and computes various statistics on the provided datasets, including demographics, 
    clinical outcomes, and intervention effects. The function performs data merging, statistical 
    tests, and outputs results for gender, age, height, weight, BMI, and pain severity.
    Data Sources:
        - Training and test datasets
        - Demographics and clinical data
        - Cluster and treatment assignment data
        - Outcomes data (RCT and EOT)
    Steps:
        1. Merge training, test, and raw data to include demographics and clinical information.
        2. Analyze gender, age, height, weight, and BMI distributions between treatment groups.
        3. Compute pain severity statistics at baseline and end-of-treatment (EOT).
        4. Compare personalized and non-personalized treatment groups.
        5. Evaluate intervention effects by therapy type (EMDR, PERT, EMI).
        6. Test significance of pain reduction and compute interaction effects.
    Outputs:
        - Counts of diagnoses and clusters
        - Statistical comparisons (e.g., t-tests, KS tests) for demographics and clinical outcomes
        - Pain severity reduction statistics and effect sizes (e.g., Cohen's d, Hedges' g)
        - Regression analysis for interaction effects between treatment and therapy type
    Notes:
        - Assumes specific column names in the datasets (e.g., 'ID', 'Hauptdiagnose', 'geschlecht', etc.).
        - Requires external libraries: pandas, numpy, scipy, statsmodels.
        - Outputs results to the console.
    Dependencies:
        - pandas (pd)
        - numpy (np)
        - scipy.stats (ks_2samp, ttest_1samp)
        - statsmodels.formula.api (smf)
        - Custom functions: hedges_g, cohens_d, compute_confidence_interval
    Raises:
        - KeyError: If required columns are missing in the datasets.
        - FileNotFoundError: If any of the input CSV files are not found.
        - ValueError: If data merging or statistical tests encounter invalid data.
    """
    # Read the data
    data_train = pd.read_csv('PERPAIN/PERPAIN_0/data_train_mpid.csv')
    data_test = pd.read_csv('PERPAIN/PERPAIN_0/data_test_mpid.csv')
    entblindung = pd.read_csv('perpain_data/entblindung_matched.csv')
    raw_data = pd.read_csv('perpain_data/final_data_100823_rct.csv', delimiter=',', encoding='unicode_escape')
    # count the hauptdignose by type
 
    # combine the data
    merged_data = pd.concat([data_train, data_test], ignore_index=True)
    # merge with raw data to get the demographics
    merged_data = pd.merge(merged_data, raw_data, on='ID', how='left')
    print(merged_data['Hauptdiagnose'].value_counts())
    # merge with entblindung to get the cluster assignment
    merged_data = pd.merge(merged_data, entblindung[['ID', 'cluster', 't']], on='ID', how='left')
    print(merged_data['cluster'].value_counts())
    # join with entblindung to get the treatment assignment
    merged_data = pd.merge(merged_data, entblindung, on='ID', how='left')
    merged_data = merged_data.reset_index(drop=True)

    t = merged_data['t_y'].values
    
    # Gender analysis
    gender = merged_data['geschlecht']
    gender_0 = sum(gender[t == 0])
    gender_1 = sum(gender[t == 1])
    print('are the groups similar', ks_2samp(gender[t == 0], gender[t == 1]))

    print("gender stats")
    # count personalized woman
    # print(len(gender_0), len(gender[t == 1]))
    # print(sum(gender_0), sum(gender[t == 1]))
    # Age analysis
    age = merged_data['alter']
    alter_0_mean = np.mean(age[t == 0])
    alter_0_std = np.std(age[t == 0])
    alter_1_mean = np.mean(age[t == 1])
    alter_1_std = np.std(age[t == 1])
    print('are the groups similar',ttest_1samp(age.dropna(), 0), ks_2samp(age[t == 0], age[t == 1]))
    print('alter mean = ', alter_0_mean, alter_0_std, alter_1_mean, alter_1_std)

    # Height analysis
    height = merged_data['groesse']
    height_0_mean = np.mean(height[t == 0])
    height_0_std = np.std(height[t == 0])
    height_1_mean = np.mean(height[t == 1])
    height_1_std = np.std(height[t == 1])
    print('are the height groups similar', ks_2samp(height[t == 0], height[t == 1]))
    print('height mean = ', height_0_mean, height_0_std, height_1_mean, height_1_std)

    # Weight analysis
    weight = merged_data['gewicht']
    weight_0_mean = np.mean(weight[t == 0])
    weight_0_std = np.std(weight[t == 0])
    weight_1_mean = np.mean(weight[t == 1])
    weight_1_std = np.std(weight[t == 1])
    print('are the weight groups similar', ks_2samp(weight[t == 0], weight[t == 1]))
    print('weight mean = ', weight_0_mean, weight_0_std, weight_1_mean, weight_1_std)
    print('personalized: ', np.sum(t))
    print('personalized', merged_data['Hauptdiagnose'][t == 1].value_counts())
    print('nonpersonalized', merged_data['Hauptdiagnose'][t == 0].value_counts())
    print(merged_data['cluster'].value_counts())
    print(merged_data['t'].value_counts())

    # BMI analysis
    bmi = weight / height * 100
    bmi_0_mean = np.mean(bmi[t == 0])
    bmi_0_std = np.std(bmi[t == 0])
    bmi_1_mean = np.mean(bmi[t == 1])
    bmi_1_std = np.std(bmi[t == 1])
    print('are the bmi groups similar', ks_2samp(bmi[t == 0], bmi[t == 1]))
    print('bmi mean = ', bmi_0_mean, bmi_0_std, bmi_1_mean, bmi_1_std)
    print('personalized: ', np.sum(t))
    print('personalized', merged_data['Hauptdiagnose'][t == 1].value_counts())
    print('nonpersonalized', merged_data['Hauptdiagnose'][t == 0].value_counts())
    print(merged_data['cluster'].value_counts())
    print(merged_data['t'].value_counts())

    # Load outcomes and merge
    outcomes_rct = pd.read_csv('perpain_data/outcomes_rct.csv')
    outcomes_eot = pd.read_csv('perpain_data/outcomes_eot.csv')
    outcomes_rct_merged = pd.merge(outcomes_rct, entblindung, on='ID', how='inner')
    outcomes_eot_merged = pd.merge(outcomes_eot, entblindung, on='ID', how='inner')

    mpid_ps_pre = outcomes_rct_merged['mpid_PS']
    mpid_ps_eot = outcomes_eot_merged['mpid_PS_eot']
    therapie = outcomes_rct_merged['therapie']

    # Overall pain severity stats
    print("baseline pain severity", np.mean(mpid_ps_pre), np.std(mpid_ps_pre))
    print("baseline pain severity personalized", np.mean(mpid_ps_pre[t == 1]), np.std(mpid_ps_pre[t == 1]))
    print("baseline pain severity nonpersonalized", np.mean(mpid_ps_pre[t == 0]), np.std(mpid_ps_pre[t == 0]))
    print("eot pain severity", np.mean(mpid_ps_pre) - np.mean(mpid_ps_eot), np.std(mpid_ps_pre - mpid_ps_eot))
    print("eot pain severity Hedges'g", hedges_g(mpid_ps_pre, mpid_ps_eot))
    print("reduction in personalized group", np.mean(mpid_ps_pre[t == 1] - mpid_ps_eot[t == 1]),
          np.std(mpid_ps_pre[t == 1] - mpid_ps_eot[t == 1]))
    print("reduction in nonpersonalized group", np.mean(mpid_ps_pre[t == 0] - mpid_ps_eot[t == 0]),
          np.std(mpid_ps_pre[t == 0] - mpid_ps_eot[t == 0]))

    diff_1 = mpid_ps_pre[t == 1] - mpid_ps_eot[t == 1]
    diff_0 = mpid_ps_pre[t == 0] - mpid_ps_eot[t == 0]
    diff = np.mean(diff_1) - np.mean(diff_0)
    print("diff = ", diff)
    print("test", ks_2samp(diff_0, diff_1))
    print("mean difference in groups")
    from utils.stats import compute_confidence_interval
    compute_confidence_interval(diff_0, diff_1)
    print("mean difference in groups Hedges'g", hedges_g(diff_0, diff_1))

    # Compute differences by intervention type
    print("\nTherapie counts:", outcomes_rct_merged['therapie'].value_counts())

    # Personalized arms
    emdr_pers_baseline = mpid_ps_pre[t == 1][therapie == "EMDR"]
    emdr_pers_eot = mpid_ps_eot[t == 1][therapie == "EMDR"]
    mead_diff_pers_emdr = np.mean(emdr_pers_baseline - emdr_pers_eot)
    std_diff_pers_emdr = np.std(emdr_pers_baseline - emdr_pers_eot)
    print("Pers EMDR", mead_diff_pers_emdr, std_diff_pers_emdr, 'cohens d', cohens_d(emdr_pers_baseline, emdr_pers_eot))

    pert_pers_baseline = mpid_ps_pre[t == 1][therapie == "PERT"]
    pert_pers_eot = mpid_ps_eot[t == 1][therapie == "PERT"]
    mead_diff_pers_pert = np.mean(pert_pers_baseline - pert_pers_eot)
    std_diff_pers_pert = np.std(pert_pers_baseline - pert_pers_eot)
    print("Pers PERT", mead_diff_pers_pert, std_diff_pers_pert, 'cohens d', cohens_d(pert_pers_baseline, pert_pers_eot))

    emi_pers_baseline = mpid_ps_pre[t == 1][therapie == "EMI"]
    emi_pers_eot = mpid_ps_eot[t == 1][therapie == "EMI"]
    mead_diff_pers_emi = np.mean(emi_pers_baseline - emi_pers_eot)
    std_diff_pers_emi = np.std(emi_pers_baseline - emi_pers_eot)
    print("Pers EMI", mead_diff_pers_emi, std_diff_pers_emi, 'cohens d', cohens_d(emi_pers_baseline, emi_pers_eot))

    # Non-personalized arms
    emdr_nonpers_baseline = mpid_ps_pre[t == 0][therapie == "EMDR"]
    emdr_nonpers_eot = mpid_ps_eot[t == 0][therapie == "EMDR"]
    mead_diff_nonpers_emdr = np.mean(emdr_nonpers_baseline - emdr_nonpers_eot)
    std_diff_nonpers_emdr = np.std(emdr_nonpers_baseline - emdr_nonpers_eot)
    print("Nonpers EMDR", mead_diff_nonpers_emdr, std_diff_nonpers_emdr, 'cohens d',
          cohens_d(emdr_nonpers_baseline, emdr_nonpers_eot))

    pert_nonpers_baseline = mpid_ps_pre[t == 0][therapie == "PERT"]
    pert_nonpers_eot = mpid_ps_eot[t == 0][therapie == "PERT"]
    mead_diff_nonpers_pert = np.mean(pert_nonpers_baseline - pert_nonpers_eot)
    std_diff_nonpers_pert = np.std(pert_nonpers_baseline - pert_nonpers_eot)
    print("Nonpers PERT", mead_diff_nonpers_pert, std_diff_nonpers_pert, 'cohens d',
          cohens_d(pert_nonpers_baseline, pert_nonpers_eot))

    emi_nonpers_baseline = mpid_ps_pre[t == 0][therapie == "EMI"]
    emi_nonpers_eot = mpid_ps_eot[t == 0][therapie == "EMI"]
    mead_diff_nonpers_emi = np.mean(emi_nonpers_baseline - emi_nonpers_eot)
    std_diff_nonpers_emi = np.std(emi_nonpers_baseline - emi_nonpers_eot)
    print("Nonpers EMI", mead_diff_nonpers_emi, std_diff_nonpers_emi, 'cohens d',
          cohens_d(emi_nonpers_baseline, emi_nonpers_eot))

    # Test significance of pain reduction vs. baseline
    print("\n=== Testing Significance of Pain Reduction vs. Baseline ===")
    emdr_pers_diff = emdr_pers_baseline - emdr_pers_eot
    pert_pers_diff = pert_pers_baseline - pert_pers_eot
    emi_pers_diff = emi_pers_baseline - emi_pers_eot
    emdr_nonpers_diff = emdr_nonpers_baseline - emdr_nonpers_eot
    pert_nonpers_diff = pert_nonpers_baseline - pert_nonpers_eot
    emi_nonpers_diff = emi_nonpers_baseline - emi_nonpers_eot

    print("Personalized EMDR:", ttest_1samp(emdr_pers_diff.dropna(), 0))
    print("Personalized PERT:", ttest_1samp(pert_pers_diff.dropna(), 0))
    print("Personalized EMI:", ttest_1samp(emi_pers_diff.dropna(), 0))
    print("Non-personalized EMDR:", ttest_1samp(emdr_nonpers_diff.dropna(), 0))
    print("Non-personalized PERT:", ttest_1samp(pert_nonpers_diff.dropna(), 0))
    print("Non-personalized EMI:", ttest_1samp(emi_nonpers_diff.dropna(), 0))

    # Interaction effect with explicit reference category
    print("\n=== Computing Interaction Effect (Reference: EMDR) ===")
    reg_data = pd.DataFrame({
        'pain_reduction': mpid_ps_pre - mpid_ps_eot,
        't': outcomes_rct_merged['t'],
        'therapie': outcomes_rct_merged['therapie']
    })
    reg_data = reg_data.dropna()
    # Explicitly set EMDR as the reference category
    model = smf.ols(
        'pain_reduction ~ t + C(therapie, Treatment(reference="EMDR")) + t:C(therapie, Treatment(reference="EMDR"))',
        data=reg_data).fit()
    print(model.summary())
    print("\nKey Interaction Results:")
    for param in model.params.index:
        if ':' in param:
            p_value = model.pvalues[param]
            print(f"{param}: Coefficient = {model.params[param]:.3f}, p-value = {p_value:.3f}")
            if p_value < 0.05:
                print(f"  -> Significant interaction (p < 0.05)")
            else:
                print(f"  -> Not significant (p >= 0.05)")