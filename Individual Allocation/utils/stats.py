import pandas as pd
import numpy as np
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_1samp, ks_2samp
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import lilliefors
from sklearn.manifold import TSNE
import math
from utils.data_processing import read_data
import scipy
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import ttest_ind


def mean_confidence_interval(data, confidence=0.99):
    """
    Calculate the mean and confidence interval for a given dataset.

    This function computes the mean and the margin of error for the confidence interval
    of the mean using the t-distribution.

    Parameters:
        data (list or array-like): The dataset for which the mean and confidence interval are calculated.
        confidence (float, optional): The confidence level for the interval. Default is 0.99 (99%).

    Returns:
        tuple: A tuple containing:
            - m (float): The mean of the dataset.
            - h (float): The margin of error for the confidence interval.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> mean_confidence_interval(data, confidence=0.95)
        (3.0, 1.96)
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h



def test_normality(data_eot, treatment):
    """
    Analyzes the normality of data by separating it based on treatment groups 
    and calculating statistics for each group.

    Args:
        data_eot (numpy.ndarray): The input data array, where each element corresponds 
            to a data point.
        treatment (numpy.ndarray): An array indicating the treatment group for each 
            data point in `data_eot`. Expected values are 0, 1, or 2.

    Returns:
        None: This function does not return a value. It performs statistical 
        calculations for each treatment group and outputs the results.
    """
    # Histogram
    data = np.squeeze(data_eot)
    treatment = np.squeeze(treatment)

    data_t0 = data[treatment == 0]
    get_statistics(data_t0, 'PERT')

    data_t1 = data[treatment == 1]
    get_statistics(data_t1, 'EMA')

    data_t2 = data[treatment == 2]
    get_statistics(data_t2, 'EMDR')
    return


def get_statistics(data, name):
    """
    Generate and display various statistical analyses and visualizations for the given data.

    Parameters:
    -----------
    data : array-like
        The input data for statistical analysis. It should be a numeric array or similar structure.
    name : str
        A name or label for the dataset, used in plot titles and printed outputs.

    Visualizations:
    ---------------
    1. Histogram with KDE (Kernel Density Estimate).
    2. Q-Q Plot to assess normality.
    3. Box Plot to visualize data distribution and outliers.

    Statistical Tests:
    ------------------
    1. Shapiro-Wilk Test:
        Tests for normality of the data.
        Outputs the test statistic and p-value.
    2. Kolmogorov-Smirnov Test:
        Compares the data distribution to a normal distribution.
        Outputs the test statistic and p-value.
    3. Anderson-Darling Test:
        Tests for normality and provides critical values and significance levels.
    4. Lilliefors Test:
        A variation of the Kolmogorov-Smirnov test for normality (requires statsmodels library).
        Outputs the test statistic and p-value.

    Returns:
    --------
    None
        The function displays plots and prints the results of statistical tests.

    Notes:
    ------
    - Ensure that the required libraries (numpy, matplotlib, seaborn, scipy.stats, and statsmodels) are installed.
    - The input data is squeezed to remove single-dimensional entries from the shape.
    """

    data = np.squeeze(data)

    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title('Histogram ' + name)
    plt.show()
    # Q-Q Plot
    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot ' + name)
    plt.show()
    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data)
    plt.title('Box Plot ' + name)
    plt.show()
    # Shapiro-Wilk Test
    shapiro_test = stats.shapiro(data)
    print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue} ", name)
    # Kolmogorov-Smirnov Test
    ks_test = stats.kstest(data, 'norm')
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_test.statistic}, p-value={ks_test.pvalue} ", name)
    # Anderson-Darling Test
    ad_test = stats.anderson(data, dist='norm')
    print(
        f"Anderson-Darling Test: Statistic={ad_test.statistic}, Critical Values={ad_test.critical_values}, Significance Level={ad_test.significance_level} ",
        name)
    # Lilliefors Test (requires statsmodels library)
    lilliefors_test = lilliefors(data)
    print(f"Lilliefors Test: Statistic={lilliefors_test[0]}, p-value={lilliefors_test[1]} ", name)
    return



def compare_distributions():
    """
    Compare distributions of data based on treatment groups and personalization.

    This function reads data, processes it to separate into different groups based on 
    treatment received (EMA, EMDR, PERT) and personalization (perso, non-perso), 
    performs t-SNE dimensionality reduction, and visualizes the results using histograms. 
    Additionally, it performs statistical tests to compare the distributions of the 
    reduced data for each group.

    Steps:
    1. Reads and processes data to separate into groups based on treatment and personalization.
    2. Performs t-SNE dimensionality reduction on the selected features.
    3. Plots histograms of the reduced data for each group.
    4. Conducts Kolmogorov-Smirnov tests to compare distributions between groups.

    Note:
    - The function assumes the existence of a CSV file named 'perpain_data/entblindung_matched.csv'.
    - The t-SNE dimensionality reduction is performed with `n_components=1` and `perplexity=10`.
    - The selected columns for analysis are hardcoded in the function.

    Outputs:
    - Plots of t-SNE reduced data for each treatment group and personalization category.
    - Results of Kolmogorov-Smirnov tests printed to the console.

    Dependencies:
    - pandas (pd)
    - numpy (np)
    - matplotlib.pyplot (plt)
    - sklearn.manifold.TSNE
    - scipy.stats.ks_2samp

    Raises:
    - KeyError: If the required columns are not present in the input data.
    - FileNotFoundError: If the CSV file 'perpain_data/entblindung_matched.csv' is not found.
    - ValueError: If t-SNE fails due to invalid input data.

    """
    label = 'mpid_PS'
    data_rct_original, outcomes_rct, outcomes_eot, t_received = read_data()
    outcomes_labels_eot = label + '_eot'
    entblindung = pd.read_csv('perpain_data/entblindung_matched.csv')
    t_perso = entblindung['t'].values
    t_perso = np.squeeze(t_perso)
    t_received = entblindung['t_recived'].values
    perso = np.where(t_perso == 1)
    received_EMA_perso = np.where(t_received[perso] == 1)
    received_EMDR_perso = np.where(t_received[perso] == 0)
    received_PERT_perso = np.where(t_received[perso] == 2)
    non_perso = np.where(t_perso == 0)
    received_EMA_non_perso = np.where(t_received[non_perso] == 1)
    received_EMDR_non_perso = np.where(t_received[non_perso] == 0)
    received_PERT_non_perso = np.where(t_received[non_perso] == 2)
    selected_cols = ['cpg_1', 'cpg_2', 'cpg_3', "ctq_20", "ctq_21", "ctq_23", "ctq_24", "ctq_27",
                     "pss_1", "pss_2", "pss_3", "pss_6", "pss_9", "pss_10",
                     "ssd12_3", "ssd12_6", "ssd12_9", "ssd12_11",
                     "wi7_3", "wi7_5", "wi7_7"]
    data = data_rct_original[selected_cols]
    data_recieved_EMA_perso = data.iloc[received_EMA_perso]
    data_recieved_EMDR_perso = data.iloc[received_EMDR_perso]
    data_recieved_PERT_perso = data.iloc[received_PERT_perso]
    data_recieved_EMA_non_perso = data.iloc[received_EMA_non_perso]
    data_recieved_EMDR_non_perso = data.iloc[received_EMDR_non_perso]
    data_recieved_PERT_non_perso = data.iloc[received_PERT_non_perso]
    tsne = TSNE(n_components=1, random_state=42, perplexity=10)
    data_recieved_EMA_perso_tsne = tsne.fit_transform(data_recieved_EMA_perso)
    data_recieved_EMDR_perso_tsne = tsne.fit_transform(data_recieved_EMDR_perso)
    data_recieved_EMDR_non_perso_tsne = tsne.fit_transform(data_recieved_EMDR_non_perso)
    data_recieved_PERT_perso_tsne = tsne.fit_transform(data_recieved_PERT_perso)
    data_recieved_PERT_non_perso_tsne = tsne.fit_transform(data_recieved_PERT_non_perso)
    data_recieved_EMA_non_perso_tsne = tsne.fit_transform(data_recieved_EMA_non_perso)
    # plot histograms
    plt.clf()
    plt.plot(data_recieved_PERT_perso_tsne, label='PERT perso')
    plt.plot(data_recieved_PERT_non_perso_tsne, label='PERT non perso')
    plt.legend(loc='upper right')
    plt.show()
    print('are distributions of test for PERT perso vs PERT non perso and same? ',
          ks_2samp(np.squeeze(data_recieved_PERT_perso_tsne), np.squeeze(data_recieved_EMDR_perso_tsne)))
    plt.plot(data_recieved_EMA_perso_tsne, label='EMA perso')
    plt.plot(data_recieved_EMA_non_perso_tsne, label='EMA non perso')
    plt.legend(loc='upper right')
    plt.show()
    print('are distributions of test for EMA perso vs EMA non perso and same? ',
          ks_2samp(np.squeeze(data_recieved_EMA_perso_tsne), np.squeeze(data_recieved_EMA_non_perso_tsne)))
    plt.plot(data_recieved_EMDR_perso_tsne, label='EMDR perso')
    plt.plot(data_recieved_EMDR_non_perso_tsne, label='EMDR non perso')
    plt.legend(loc='upper right')
    plt.show()
    print('are distributions of test for EMDR perso vs EMDR non perso and same? ',
          ks_2samp(np.squeeze(data_recieved_EMDR_perso_tsne), np.squeeze(data_recieved_EMDR_non_perso_tsne)))
    print(data_recieved_PERT_non_perso)



def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and confidence interval for a given dataset.

    Parameters:
        data (array-like): The input data for which the mean and confidence interval are calculated.
        confidence (float, optional): The confidence level for the interval. Default is 0.95 (95%).

    Returns:
        tuple: A tuple containing:
            - m (float): The mean of the data.
            - h (float): The margin of error for the confidence interval.

    Notes:
        This function assumes that the data follows a normal distribution.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def compute_confidence_interval(data1, data2, confidence=0.95):
    """
    Computes the confidence interval for the difference in means between two datasets.

    Parameters:
        data1 (array-like): The first dataset, a sequence of numerical values.
        data2 (array-like): The second dataset, a sequence of numerical values.
        confidence (float, optional): The confidence level for the interval. Default is 0.95 (95%).

    Returns:
        None: The function prints the following results:
            - Mean Difference: The difference between the means of the two datasets.
            - Standard Error: The standard error of the difference in means.
            - Degrees of Freedom: The effective degrees of freedom for the t-distribution.
            - T-Critical: The critical t-value for the specified confidence level.
            - Confidence Interval: The lower and upper bounds of the confidence interval.

    Notes:
        - The function assumes the input datasets are independent and may have unequal variances.
        - The Welch-Satterthwaite equation is used to compute the degrees of freedom.
        - The confidence interval is calculated using the t-distribution.

    Example:
        >>> data1 = [1.2, 2.3, 3.1, 4.0]
        >>> data2 = [1.8, 2.5, 3.0, 3.7]
        >>> compute_confidence_interval(data1, data2)
        Mean Difference: -0.3500
        Standard Error: 0.2450
        Degrees of Freedom: 6.78
        T-Critical: 2.4470
        Confidence Interval: (-0.8500, 0.1500)
    """
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    std1 = np.std(data1)
    std2 = np.std(data2)
    n1 = len(data1)
    n2 = len(data2)
    confidence = 0.95
    # Step 1: Compute the difference in means
    mean_difference = mean1 - mean2

    # Step 2: Compute the standard error of the difference
    se_difference = math.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

    # Step 3: Compute degrees of freedom using the Welch-Satterthwaite equation
    df = ((std1 ** 2 / n1 + std2 ** 2 / n2) ** 2 /
          (((std1 ** 2 / n1) ** 2 / (n1 - 1)) + ((std2 ** 2 / n2) ** 2 / (n2 - 1))))

    # Step 4: Find the critical t-value for the confidence level
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha / 2, df)

    # Step 5: Compute the confidence interval
    margin_of_error = t_critical * se_difference
    ci_lower = mean_difference - margin_of_error
    ci_upper = mean_difference + margin_of_error

    # Output results
    print(f"Mean Difference: {mean_difference:.4f}")
    print(f"Standard Error: {se_difference:.4f}")
    print(f"Degrees of Freedom: {df:.2f}")
    print(f"T-Critical: {t_critical:.4f}")
    print(f"Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")


def hedges_g(data1, data2):
    """
    Calculate Hedges' g, an effect size measure that corrects Cohen's d for small sample sizes.

    Hedges' g is used to quantify the difference between two groups in terms of standard deviations,
    while accounting for small sample size bias.

    Parameters:
        data1 (array-like): The first dataset (e.g., a list or numpy array of numerical values).
        data2 (array-like): The second dataset (e.g., a list or numpy array of numerical values).

    Returns:
        float: The Hedges' g value, representing the standardized mean difference between the two datasets.

    Notes:
        - This function assumes that the input datasets are independent and normally distributed.
        - The pooled standard deviation is used to calculate the effect size.
        - A correction factor is applied to Cohen's d to account for small sample sizes.
    """
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    sd1 = np.std(data1)
    sd2 = np.std(data2)
    n1 = len(data1)
    n2 = len(data2)
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

    # Calculate raw effect size (Cohen's d)
    cohen_d = (mean1 - mean2) / pooled_sd

    # Apply correction for small sample sizes
    correction = 1 - (3 / (4 * (n1 + n2) - 9))

    # Calculate Hedges' g
    hedges_g_value = cohen_d * correction
    return hedges_g_value


def cohens_d(data1, data2):
    """
    Calculate Cohen's d, a measure of effect size that quantifies the difference 
    between two means in terms of standard deviation.

    Parameters:
        data1 (array-like): The first dataset (e.g., a list or numpy array).
        data2 (array-like): The second dataset (e.g., a list or numpy array).

    Returns:
        float: The calculated Cohen's d value.

    Notes:
        - Cohen's d is calculated using the pooled standard deviation of the two datasets.
        - A positive value indicates that the mean of `data1` is greater than the mean of `data2`.
        - A negative value indicates that the mean of `data1` is less than the mean of `data2`.
        - Assumes that the two datasets are independent and approximately normally distributed.
    """
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    sd1 = np.std(data1)
    sd2 = np.std(data2)
    n1 = len(data1)
    n2 = len(data2)
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_sd
    return d


def get_statistics_for_all_variables():
    """
    Computes and analyzes statistics for various variables across multiple datasets.
    This function reads and merges data from multiple CSV files, calculates BMI, 
    performs statistical tests (Kolmogorov-Smirnov test, independent t-test), and 
    computes Cohen's d effect size for each variable. It also checks for missing 
    data and outputs the results to a CSV file. Additionally, it fits a linear 
    model to analyze the interaction effect between treatment and personalization.
    Outputs:
        - A CSV file ("stats.csv") containing statistical results for each variable.
        - Prints missing data summary and linear model summary to the console.
    Data Sources:
        - 'perpain_data/outcomes_rct.csv': Outcome data.
        - 'PERPAIN/PERPAIN_0/data_train_mpid.csv': Training data.
        - 'PERPAIN/PERPAIN_0/data_test_mpid.csv': Test data.
        - 'PERPAIN/entblindung.csv': Personalization data.
    Statistical Tests:
        - Kolmogorov-Smirnov test for distribution comparison.
        - Independent t-test for mean comparison.
        - Cohen's d for effect size.
    Linear Model:
        - Analyzes the interaction effect between treatment and personalization 
          on MPI_D reduction.
    Returns:
        None
    Notes:
        - Ensure all required CSV files are present in the specified paths.
        - Missing data is handled by dropping rows with NaN values for each variable.
        - The function assumes specific column names in the datasets.
    """
    outcomes_to_quest = {
        'Demographie': ['alter', 'geschlecht', 'groesse', 'gewicht', 'belastungen', 'BMI'],
        'EPIC': ['epic_1', 'epic_2a_summer', 'epic_2a_winter', 'epic_2b_summer',
                 'epic_2b_winter', 'epic_2c_summer', 'epic_2c_winter', 'epic_2d_summer',
                 'epic_2d_winter', 'epic_2e_summer', 'epic_2e_winter', 'epic_2f_summer',
                 'epic_2f_winter', 'epic_3', 'epic_3_1'],
        'BSI': ['bsi_somat', 'bsi_zwan', 'bsi_unsicher', 'bsi_dipress', 'bsi_angst',
                'bsi_agress', 'bsi_fph', 'bsi_paranoid', 'bsi_psycho', 'bsi_zusatz',
                'bsi_gsi'],
        'CPG': ['cpg', 'cpg_beeintraechtigung', 'cpg_intensitaet'],
        'CQR5': ['cqr5_low', 'cqr5_high', 'cqr5_bin'],
        'MPID': ['mpid_PS', 'mpid_I', 'mpid_LC', 'mpid_AD', 'mpid_S', 'mpid_PR',
                 'mpid_SR', 'mpid_DR', 'mpi_sa', 'mpi_aih', 'mpi_aah', 'mpi_GA'],
        'ODI': ['odi'],
        'SES': ['ses_affective', 'ses_sensoric'],
        'SSD12': ['ssd12', 'ssd12_cognitive', 'ssd12_affective', 'ssd12_behavioral'],
        'SSS8': ['sss8'],
        'TICS': ['tics_uebe', 'tics_soue', 'tics_erdr', 'tics_unzu', 'tics_uefo',
                 'tics_mang', 'tics_sozs', 'tics_sozi', 'tics_sorg', 'tics_sscs'],
        'WI7': ['wi7', 'wi7_illworr', 'wi7_illconv'],
        'WOMAC': ['womac_total', 'womac_pain', 'womac_stiffness', 'womac_physical_function'],
        'WPI': ['wpi'],
        'PANAS': ['sum_panaspa', 'sum_panasna'],
        'HADS': ['hads_a', 'hads_d'],
        'PSQI': ['psqi'],
        'KLL': ['sum_kll'],
        'CTQ': ['sum_ctq', 'ctq_em', 'ctq_km', 'ctq_sm', 'ctq_ev', 'ctq_kv', 'ctq_min'],
        'FSS': ['fss'],
        'PSS': ['pss', 'pss_wh', 'pss_ws'],
        'RS11': ['rs11'],
        'SF12': ['KSK12', 'PSK12'],
        'FABQ': ['fabq', 'fabq_wc', 'fabq_pw', 'fabq_pa'],
        'FPQ': ['fpq_total', 'fpq_severe_pain', 'fpq_medical_pain', 'fpq_minor_pain'],
        'PRSS': ['mean_prss', 'prss_catastrophizing', 'prss_coping']
    }

    # Read and merge data
    # raw_data_rct = pd.read_csv('perpain_data/data_rct.csv', delimiter=',', encoding='unicode_escape')
    raw_data_outcomes = pd.read_csv('perpain_data/outcomes_rct.csv', delimiter=',', encoding='unicode_escape')
    
    raw_data_rct_train = pd.read_csv('PERPAIN/PERPAIN_0/data_train_mpid.csv')
    raw_data_rct_test = pd.read_csv('PERPAIN/PERPAIN_0/data_test_mpid.csv')
    raw_data_rct = pd.concat([raw_data_rct_train, raw_data_rct_test], ignore_index=True)

    
    #raw_data_rct = raw_data_rct.drop(columns=['ID'])
    raw_data_rct['BMI'] = raw_data_rct['gewicht'] / ((raw_data_rct['groesse']*0.01) ** 2)
    raw_data_combined = pd.merge(raw_data_rct, raw_data_outcomes,  on='ID', how='left')

    entblindung = pd.read_csv('PERPAIN/entblindung.csv')
    t = entblindung["t"]
    merged_data = pd.merge(raw_data_combined, entblindung, on='ID', how='left')

    # Initialize output DataFrame
    data_out = pd.DataFrame(columns=["Score", "Personalized", "Non-Personalized", "KS p-value", "t-statistic", "df", "t-test p-value", "Cohen's d"])

    # Check missing data
    print("\n=== Missing Data Summary ===")
    for key in outcomes_to_quest:
        for scale in outcomes_to_quest[key]:
            missing_count = merged_data[scale].isna().sum()
            if missing_count > 0:
                print(f"{scale}: {missing_count} missing values")

    i = 0
    for key in outcomes_to_quest:
        for scale in outcomes_to_quest[key]:
            # Drop missing values for this scale
            values = merged_data[scale].dropna()
            t_valid = t[merged_data[scale].notna()]
            values_0 = values[t_valid == 0]
            values_1 = values[t_valid == 1]

            # Compute statistics
            values_0_mean = np.mean(values_0) if len(values_0) > 0 else np.nan
            values_0_std = np.std(values_0, ddof=1) if len(values_0) > 1 else np.nan
            values_1_mean = np.mean(values_1) if len(values_1) > 0 else np.nan
            values_1_std = np.std(values_1, ddof=1) if len(values_1) > 1 else np.nan
            ks_p_val = ks_2samp(values_0, values_1).pvalue if len(values_0) > 0 and len(values_1) > 0 else np.nan

            # Independent t-test
            if len(values_0) > 1 and len(values_1) > 1:
                t_stat, t_p_val = ttest_ind(values_0, values_1, equal_var=True)
                df = len(values_0) + len(values_1) - 2
                d = cohens_d(values_0, values_1)
            else:
                t_stat, t_p_val, df, d = np.nan, np.nan, np.nan, np.nan

            # Format output
            new_row = pd.DataFrame({
                'Score': scale,
                'Personalized': f"{values_0_mean:.2f} ({values_0_std:.2f})" if not np.isnan(values_0_mean) else "NaN",
                'Non-Personalized': f"{values_1_mean:.2f} ({values_1_std:.2f})" if not np.isnan(values_1_mean) else "NaN",
                'KS p-value': round(ks_p_val, 2) if not np.isnan(ks_p_val) else "NaN",
                't-statistic': round(t_stat, 2) if not np.isnan(t_stat) else "NaN",
                'df': int(df) if not np.isnan(df) else "NaN",
                't-test p-value': round(t_p_val, 3) if not np.isnan(t_p_val) else "NaN",
                "Cohen's d": round(d, 2) if not np.isnan(d) else "NaN"
            }, index=[i])

            data_out = pd.concat([data_out, new_row], ignore_index=True)
            i += 1

    # Save results
    data_out.to_csv("stats.csv", index=False)
    print(data_out)

    # compute the interraction effect
    mpid_ps_pre = merged_data['mpid_PS']
    personalization = t
    treatment = merged_data['t_recived']

    data = pd.DataFrame()
    data['MPI_D_reduction'] = mpid_ps_pre
    data['Treatment'] = treatment
    data['Personalization'] = personalization
    # Fit linear model with interaction term
    model = smf.ols('MPI_D_reduction ~ C(Treatment) * C(Personalization)', data=data).fit()
    print(model.summary())

def find_means_and_std(data, model, n_iterations):
    """
    Computes the mean and standard deviation of predictions for each output dimension
    across multiple iterations of a model's predictions.

    Args:
        data (numpy.ndarray): Input data for the model, with shape (n_subjects, ...).
        model (callable): A callable model that takes `data` as input and returns predictions.
                          The model should support a `training` argument for stochastic behavior.
        n_iterations (int): Number of iterations to perform for generating predictions.

    Returns:
        tuple: A tuple containing six lists:
            - y0_pred_mean (list): Mean predictions for the first output dimension.
            - y0_pred_std (list): Standard deviations of predictions for the first output dimension.
            - y1_pred_mean (list): Mean predictions for the second output dimension.
            - y1_pred_std (list): Standard deviations of predictions for the second output dimension.
            - y2_pred_mean (list): Mean predictions for the third output dimension.
            - y2_pred_std (list): Standard deviations of predictions for the third output dimension.

    Notes:
        - The model is expected to return predictions with shape (n_subjects, 3) for each iteration.
        - The function assumes the output has three dimensions (e.g., y0, y1, y2).
        - The predictions are aggregated across `n_iterations` to compute the mean and standard deviation
          for each subject and each output dimension.
    """
    # add the train set performance check
    predictions = np.zeros((n_iterations, data.shape[0], 3))
    for i in range(n_iterations):
        preds = model(data, training=True).numpy()
        y0_preds = preds[:, 0]
        y1_preds = preds[:, 1]
        y2_preds = preds[:, 2]
        predictions[i, :, 0] = np.squeeze(y0_preds)
        predictions[i, :, 1] = np.squeeze(y1_preds)
        predictions[i, :, 2] = np.squeeze(y2_preds)
    # check the prediction quality
    y0_pred_mean = []
    y1_pred_mean = []
    y2_pred_mean = []
    y0_pred_std = []
    y1_pred_std = []
    y2_pred_std = []
    predictions_rct_T = predictions.T
    preds_0 = predictions_rct_T[0]
    preds_1 = predictions_rct_T[1]
    preds_2 = predictions_rct_T[2]
    n_subjects = data.shape[0]
    for i in range(n_subjects):
        mean_0 = np.mean(preds_0[i])
        std_0 = np.std(preds_0[i])
        mean_1 = np.mean(preds_1[i])
        std_1 = np.std(preds_1[i])
        mean_2 = np.mean(preds_2[i])
        std_2 = np.std(preds_2[i])

        y0_pred_mean.append(mean_0)
        y1_pred_mean.append(mean_1)
        y2_pred_mean.append(mean_2)

        y0_pred_std.append(std_0)
        y1_pred_std.append(std_1)
        y2_pred_std.append(std_2)
    return y0_pred_mean, y0_pred_std, y1_pred_mean, y1_pred_std, y2_pred_mean, y2_pred_std



def find_means_and_std_bayesian(data, model, scaler=None):
    """
    Computes the mean and standard deviation of predictions from a Bayesian model.

    Args:
        data (numpy.ndarray): Input data for the model. Expected to be a 2D array.
        model (callable): A Bayesian model that takes `data` as input and returns predictions.
                          The predictions are expected to be a 2D array where the first row
                          contains the means and the second row contains the standard deviations.
        scaler (sklearn.preprocessing.StandardScaler, optional): A scaler object used to inverse
                          transform the predictions. If provided, the predictions will be scaled
                          back to their original range.

    Returns:
        tuple: A tuple containing:
            - y0_pred_mean (numpy.ndarray): Mean predictions for the first segment of the data.
            - y0_pred_std (numpy.ndarray): Standard deviations for the first segment of the data.
            - y1_pred_mean (numpy.ndarray): Mean predictions for the second segment of the data.
            - y1_pred_std (numpy.ndarray): Standard deviations for the second segment of the data.
            - y2_pred_mean (numpy.ndarray): Mean predictions for the third segment of the data.
            - y2_pred_std (numpy.ndarray): Standard deviations for the third segment of the data.
    """
    preds = model(data, training=False).numpy()
    if scaler is not None:
        y0_pred_mean = scaler.inverse_transform(preds[0, :data.shape[0]].reshape(-1, 1)).reshape(-1)
        y1_pred_mean = scaler.inverse_transform(preds[0, data.shape[0]:2 * data.shape[0]].reshape(-1, 1)).reshape(-1)
        y2_pred_mean = scaler.inverse_transform(preds[0, 2 * data.shape[0]:].reshape(-1, 1)).reshape(-1)
    else:
        y0_pred_mean = preds[0, :data.shape[0]]
        y1_pred_mean = preds[0, data.shape[0]:2 * data.shape[0]]
        y2_pred_mean = preds[0, 2 * data.shape[0]:]

    y0_pred_std = preds[1, :data.shape[0]]
    y1_pred_std = preds[1, data.shape[0]:2 * data.shape[0]]
    y2_pred_std = preds[1, 2 * data.shape[0]:]
    return y0_pred_mean, y0_pred_std, y1_pred_mean, y1_pred_std, y2_pred_mean, y2_pred_std

def compute_mse(y0, y1, y2, y_baseline, y_eot, t_test, train = False):
    """
    Compute the mean squared error (MSE) between observed effects 
    and predicted effects for multi-treatment settings.

    Parameters
    ----------
    y0, y1, y2 : np.ndarray
        Predicted outcomes under treatments 0, 1, and 2.
    y_baseline : np.ndarray
        Baseline outcomes.
    y_eot : np.ndarray
        Observed outcomes under the assigned treatment.
    t_test : np.ndarray
        Assigned treatments (values in {0,1,2}).

    Returns
    -------
    float
        Mean squared error between observed and predicted effects.
    """
    # Observed (real) effect
    y_original_effect = y_eot - y_baseline

    # Predicted effects under each treatment
    y0_effect = y0 - y_baseline
    y1_effect = y1 - y_baseline
    y2_effect = y2 - y_baseline

    if not train:
       print("Average effect for treatment 0:", np.mean(y0_effect), np.std(y0_effect))
       print("Average effect for treatment 1:", np.mean(y1_effect), np.std(y1_effect))
       print("Average effect for treatment 2:", np.mean(y2_effect), np.std(y2_effect))

    # Stack effects: shape [n_samples, 3]
    effect = np.stack([y0_effect, y1_effect, y2_effect], axis=1)

    # Pick the predicted effect corresponding to the assigned treatment
    predicted_effect = effect[np.arange(len(t_test)), t_test]

    # Compute MSE
    return mean_squared_error(y_original_effect, predicted_effect)


