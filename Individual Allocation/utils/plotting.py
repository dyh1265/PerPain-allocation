from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from utils.stats import find_means_and_std_bayesian
from utils.stats import mean_confidence_interval, cohens_d, ks_2samp
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
from sklearn.metrics import mean_squared_error

def plot_proposed_treatments(data, y_real, y_baseline, t, model, model_name, scaler=None, mode='train'):
    """
    Plots the proposed treatments versus the original treatments for a set of patients, 
    based on the predicted and real treatment effects.

    Args:
        data (np.ndarray): Input data for the model.
        y_real (np.ndarray): Real treatment outcomes for the patients.
        y_baseline (np.ndarray): Baseline outcomes for the patients.
        t (np.ndarray): Original treatment assignments for the patients.
        model (object): Trained Bayesian model used for predictions.
        model_name (str): Name of the model, used for labeling the plot.
        scaler (object, optional): Scaler object for data preprocessing. Defaults to None.
        mode (str, optional): Mode of operation (e.g., 'train', 'test'). Defaults to 'train'.

    Returns:
        None: The function saves the plot as a PNG file and does not return any value.

    Notes:
        - The function computes the predicted treatment effects using the Bayesian model.
        - Patients are sorted by the predicted effect of the proposed treatments.
        - The plot includes bars for both the proposed and original treatments, 
          with distinct color schemes for each treatment type.
        - The plot is saved as a PNG file with a filename that includes the model name and mode.
    """
    # Compute predictions from the Bayesian model
    y0_pred_mean, y0_pred_std, y1_pred_mean, y1_pred_std, y2_pred_mean, y2_pred_std = find_means_and_std_bayesian(data, model, scaler)
    t = np.squeeze(t)
    preds = np.stack([y0_pred_mean, y1_pred_mean, y2_pred_mean], axis=1)
    preds_std = np.stack([y0_pred_std, y1_pred_std, y2_pred_std], axis=1)

    # Compute effects
    pred_effects = preds - y_baseline.reshape(-1, 1)
    real_effects = y_real.reshape(-1, 1) - y_baseline.reshape(-1, 1)
    real_effects = np.squeeze(real_effects)

    # Proposed treatments (minimum predicted effect)
    new_treatments = np.argmin(pred_effects, axis=1)
    new_std = preds_std[np.arange(len(new_treatments)), new_treatments]
    print(f"Standard deviations of proposed treatments: {new_std}")
    new_effect = np.min(pred_effects, axis=1)

    # Sort patients by predicted effect
    sorted_idx = np.argsort(new_effect)
    new_effect = new_effect[sorted_idx]
    real_effects = real_effects[sorted_idx]
    t_sorted = t[sorted_idx]
    new_treatments_sorted = new_treatments[sorted_idx]

    # Define consistent color scheme
    original_colors = {
        0: '#054A86',  # EDTT - dark blue
        1: '#70BEB7',  # EMDI - teal
        2: '#8E3A71'   # PERT - deep magenta
    }
    proposed_colors = {
        0: '#A3C1D9',  # EDTT - light blue
        1: '#C1E6E2',  # EMDI - pale teal
        2: '#D9A3C1'   # PERT - light pinkish-purple
    }
    treatment_names = {0: 'EDTT', 1: 'EMDI', 2: 'PERT'}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(new_effect))
    width = 0.4

    # Plot bars for proposed treatments (colored by proposed treatment type)
    for i, (eff, trt) in enumerate(zip(new_effect, new_treatments_sorted)):
        ax.bar(x[i] - width/2, -eff, width=width, color=proposed_colors[trt])

    # Plot bars for original treatments (colored by original treatment type)
    for i, (eff, trt) in enumerate(zip(real_effects, t_sorted)):
        ax.bar(x[i] + width/2, -eff, width=width, color=original_colors[trt])
        print(trt)
    # Legend
    legend_elements = [
        Patch(facecolor=proposed_colors[0], label='Proposed EDTT'),
        Patch(facecolor=proposed_colors[1], label='Proposed EMDI'),
        Patch(facecolor=proposed_colors[2], label='Proposed PERT'),
        Patch(facecolor=original_colors[0], label='Original EDTT'),
        Patch(facecolor=original_colors[1], label='Original EMDI'),
        Patch(facecolor=original_colors[2], label='Original PERT'),
    ]
    ax.legend(handles=legend_elements, title='Treatments', ncol=2)

    # Labels and title
    ax.set_xlabel('Patients (sorted by predicted effect)')
    ax.set_ylabel('Effect')
    ax.set_title(f'Proposed vs Original Treatments ({model_name}, {mode})')
    plt.tight_layout()

    # Save and close
    filename = f'effects_all_proposed_vs_original_{model_name}_{mode}.png'
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filename}")
    print(t_sorted)


def print_outcomes(y0, y1, y2, y_eot, y_baseline, std0, std1, std2, t_test, name, sorted_idx=None):
    """
    Generate a comparative plot of treatment effects with uncertainty bands and perform statistical analysis.

    Parameters:
    -----------
    y0 : np.ndarray
        Predicted outcomes for treatment group 0.
    y1 : np.ndarray
        Predicted outcomes for treatment group 1.
    y2 : np.ndarray
        Predicted outcomes for treatment group 2.
    y_eot : np.ndarray
        Observed end-of-treatment outcomes.
    y_baseline : np.ndarray
        Baseline outcomes for all patients.
    std0 : np.ndarray
        Standard deviations for treatment group 0 predictions.
    std1 : np.ndarray
        Standard deviations for treatment group 1 predictions.
    std2 : np.ndarray
        Standard deviations for treatment group 2 predictions.
    t_test : np.ndarray
        Array indicating the treatment group assignments for the test data.
    name : str
        Name used for the plot title and saved file.
    sorted_idx : np.ndarray, optional
        Indices for sorting the data. If None, data is sorted by the effect of treatment group 0.

    Returns:
    --------
    float
        Mean squared error (MSE) between the observed and predicted outcomes for the original treatment assignments.

    Notes:
    ------
    - The function generates a plot comparing the observed and predicted treatment effects, including uncertainty bands.
    - Statistical tests are performed to compare the distributions of observed and predicted outcomes:
        - Kolmogorov-Smirnov test for distribution similarity.
        - Mann-Whitney U test for differences between distributions.
        - Wilcoxon signed-rank test for paired samples.
    - Confidence intervals and Cohen's d effect sizes are calculated for the comparisons.
    - The plot is saved as a PNG file with the given name.
    """


    y_original_effect = y_eot - y_baseline
    y0_effect = y0 - y_baseline
    # sort by y0_effect
    if sorted_idx is None:
        sorted_idx = np.argsort(-y0_effect)

    t_test = np.squeeze(t_test)

    t_test = t_test[sorted_idx]

    t0 = np.squeeze(np.where(t_test == 0))
    t1 = np.squeeze(np.where(t_test == 1))
    t2 = np.squeeze(np.where(t_test == 2))

    y_original_effect = y_original_effect[sorted_idx]

    y0 = y0[sorted_idx]
    y1 = y1[sorted_idx]
    y2 = y2[sorted_idx]
    y_baseline = y_baseline[sorted_idx]

    y0_effect = y0 - y_baseline
    y1_effect = y1 - y_baseline
    y2_effect = y2 - y_baseline

    std0 = std0[sorted_idx]
    std1 = std1[sorted_idx]
    std2 = std2[sorted_idx]

    x = np.arange(0, len(y0_effect))

    # Create the uncertainty bands
    y0_upper = y0_effect + std0
    y0_lower = y0_effect - std0

    y1_upper = y1_effect + std1
    y1_lower = y1_effect - std1

    y2_upper = y2_effect + std2
    y2_lower = y2_effect - std2


    # Create a figure and an axis
    fig, ax = plt.subplots()
    ax.plot(t0, -y_original_effect[t0], 'o', color='lime', label='EMDR RCT (P)')
    ax.plot(t1, -y_original_effect[t1], 'o', color='yellow', label='EMA RCT (EM)')
    ax.plot(t2, -y_original_effect[t2], 'o', color='cyan', label='PERT RCT (ER)')
    # Plot the data
    ax.plot(x, -y0_effect, label='EMDR pred.', color='darkgreen')
    ax.plot(x, -y1_effect, label='EMA pred.',  color='darkorange')
    ax.plot(x, -y2_effect, label='PERT pred.',  color='darkblue')
    # ax.plot(x, y_original_effect, label='RCT', marker='o', color='red')
    # Step 4: Label the points
    mapping = {
        0: "ER",
        1: "EM",
        2: "P"
    }
    labels = [mapping[num] for num in np.squeeze(t_test)]

    for i, label in enumerate(labels):
        ax.text(x[i], -y_original_effect[i], label, fontsize=12, ha='right')

    # Fill between the upper and lower uncertainty bands
    ax.fill_between(x, -y0_lower, -y0_upper, color='green', alpha=0.1, linewidth=2)
    # Add a legend
    ax.legend()
    # Show the plot
    fig.show()
    ax.fill_between(x, -y2_lower, -y2_upper, color='b', alpha=0.1, linewidth=2)

    # Adding labels and title
    ax.set_xlabel('Patients')
    ax.set_ylabel('Effect')
    ax.set_title('Plot with Uncertainty Band ' + name)

    fig.show()
    fig.savefig(name + '.png')
    effect = np.concatenate([np.expand_dims(y0_effect, axis=-1), np.expand_dims(y1_effect, axis=-1),
                             np.expand_dims(y2_effect, axis=1)], axis=1)
    new_treatments = np.argmin(effect, axis=1)
    new_effect = np.min(effect, axis=1)

    mean_original = np.mean(y_original_effect)
    mean_predicted_new = np.mean(new_effect)

    std_original = np.std(y_original_effect)
    std_new = np.std(new_effect)
    mean_predicted_new, ce_predicted_new = mean_confidence_interval(new_effect)
    print('real difference in means:', mean_original)
    print('predicted difference in means:', mean_predicted_new)
    print('real std:', std_original)
    print('predicted std:', std_new)
    print('new treatments', new_treatments)
    print('old treatments', np.squeeze(t_test))

    predicted_original = np.asarray([effect[i, t_test[i]] for i in range(len(t_test))])
    predicted_original = np.squeeze(predicted_original)

    print('are distributions of predictions and real same? ', ks_2samp(y_original_effect, predicted_original))
    print('are distributions of new outcomes and real are different? ', ks_2samp(y_original_effect, new_effect))
    print('mse', mean_squared_error(y_original_effect, predicted_original))
    # Mann-Whitney U Test

    u_statistic, p_value = mannwhitneyu(y_original_effect, predicted_original)
    print(f"Mann-Whitney U Test: U-statistic={u_statistic}, p-value={p_value}")

    # Wilcoxon Signed-Rank Test (paired samples)
    w_statistic, p_value = wilcoxon(y_original_effect, new_effect)

    print(f"Wilcoxon Test: W-statistic={w_statistic}, p-value={p_value}")
    # check if new assignment is significantly different from the original
    print('are distributions of new outcomes and real are for EDTT ? ', ks_2samp(y_original_effect, y0_effect), cohens_d(y_original_effect, y0_effect))
    print('are distributions of new outcomes and real are for EMDI ? ', ks_2samp(y_original_effect, y1_effect), cohens_d(y_original_effect, y1_effect))
    print('are distributions of new outcomes and real are for PERT ? ', ks_2samp(y_original_effect, y2_effect), cohens_d(y_original_effect, y2_effect))



    return mean_squared_error(y_original_effect, predicted_original)

def plot_outcomes(x, y_eot, y_baseline, t, model, model_name, scaler=None, mode='test'):
    """
    Generates and saves three plots with uncertainty bands for different treatment groups (EDTT, EMDI, PERT).
    Parameters:
        x (numpy.ndarray): Input features for the model.
        y_eot (numpy.ndarray): End-of-treatment outcomes for the patients.
        y_baseline (numpy.ndarray): Baseline outcomes for the patients.
        t (numpy.ndarray): Treatment assignment array (0 for EDTT, 1 for EMDI, 2 for PERT).
        model (object): Trained model used to predict outcomes.
        model_name (str): Name of the model, used for saving the plots.
        scaler (object, optional): Scaler object for inverse transforming the predictions. Defaults to None.
        mode (str, optional): Mode of operation (e.g., 'test', 'train'). Used in plot titles and filenames. Defaults to 'test'.
    Returns:
        None: The function generates and saves plots as PNG files.
    Notes:
        - The function creates three separate plots for each treatment group:
          1. EDTT (t == 0) with dark blue color scheme.
          2. EMDI (t == 1) with teal color scheme.
          3. PERT (t == 2) with deep magenta color scheme.
        - Each plot includes:
          - Original effects (RCT) as points.
          - Predicted effects as a line.
          - Uncertainty bands around the predicted effects.
        - The plots are saved as PNG files with filenames formatted as '<model_name>_<treatment>_<mode>.png'.
    """
    (y0, std0, y1, std1, y2, std2) = find_means_and_std_bayesian(x, model)
    if scaler is not None:
        y0 = scaler.inverse_transform(y0.reshape(-1, 1)).reshape(-1)
        y1 = scaler.inverse_transform(y1.reshape(-1, 1)).reshape(-1)
        y2 = scaler.inverse_transform(y2.reshape(-1, 1)).reshape(-1)


    y_original_effect = np.squeeze(y_eot) - y_baseline
    y0_effect = y0 - y_baseline
    y1_effect = y1 - y_baseline
    y2_effect = y2 - y_baseline

    sorted_idx0 = np.argsort(-y0_effect)
    t = np.squeeze(t)

    # Plot 1: EDTT (t == 0) - with #054A86
    t0_sorted = t[sorted_idx0]
    t0 = np.squeeze(np.where(t0_sorted == 0))
    y_original_effect_0 = y_original_effect[sorted_idx0]
    y0 = y0[sorted_idx0]
    y_baseline_0 = y_baseline[sorted_idx0]
    y0_effect = y0 - y_baseline_0
    std0 = std0[sorted_idx0]
    x0 = np.arange(0, len(y0_effect))
    y0_upper = y0_effect + std0
    y0_lower = y0_effect - std0

    fig0 = plt.figure(figsize=(8, 8))
    plt.plot(t0, -y_original_effect_0[t0], 'o', color='#A3C1D9', label='EDTT RCT')  # Light blue for points
    plt.plot(x0, -y0_effect, label='EDTT pred.', color='#054A86')  # Dark blue for line
    y_original_effect0 = y_original_effect_0[t0]
  
    plt.fill_between(x0, -y0_lower, -y0_upper, color='#054A86', alpha=0.1, linewidth=2)  # Dark blue for band
    plt.legend()
    plt.xlabel('Patients')
    plt.ylabel('Effect')
    plt.title('Plot with Uncertainty Band ' + model_name + ' EDTT ' + mode)
    plt.tight_layout()
    plt.show()
    fig0.savefig(model_name + '_EDTT_' + mode + '.png')
    plt.close(fig0)

    # Plot 2: EMDI (t == 1) - with #70BEB7
    t1_sorted = t[sorted_idx0]
    t1 = np.squeeze(np.where(t1_sorted == 1))
    y_original_effect_1 = y_original_effect[sorted_idx0]
    y1 = y1[sorted_idx0]
    y_baseline_1 = y_baseline[sorted_idx0]
    y1_effect = y1 - y_baseline_1
    std1 = std1[sorted_idx0]
    x1 = np.arange(0, len(y1_effect))
    y1_upper = y1_effect + std1
    y1_lower = y1_effect - std1

    fig1 = plt.figure(figsize=(8, 8))
    plt.plot(t1, -y_original_effect_1[t1], 'o', color='#C1E6E2', label='EMDI RCT')  # Pale teal for points
    plt.plot(x1, -y1_effect, label='EMDI pred.', color='#70BEB7')  # Teal for line

    plt.ylim(-4, 6)
    plt.fill_between(x1, -y1_lower, -y1_upper, color='#70BEB7', alpha=0.1, linewidth=2)  # Teal for band
    plt.legend()
    plt.xlabel('Patients')
    plt.ylabel('Effect')
    plt.title('Plot with Uncertainty Band ' + model_name + ' EMDI ' + mode)
    plt.tight_layout()
    plt.show()
    fig1.savefig(model_name + '_EMDI_' + mode + '.png')
    plt.close(fig1)

    # Plot 3: PERT (t == 2) - with #8E3A71
    t2_sorted = t[sorted_idx0]
    t2 = np.squeeze(np.where(t2_sorted == 2))
    y_original_effect_2 = y_original_effect[sorted_idx0]
    y2 = y2[sorted_idx0]
    y_baseline_2 = y_baseline[sorted_idx0]
    y2_effect = y2 - y_baseline_2
    std2 = std2[sorted_idx0]
    x2 = np.arange(0, len(y2_effect))
    y2_upper = y2_effect + std2
    y2_lower = y2_effect - std2

    fig2 = plt.figure(figsize=(8, 8))
    plt.plot(t2, -y_original_effect_2[t2], 'o', color='#D9A3C1', label='PERT RCT')  # Light purple for points
    plt.plot(x2, -y2_effect, label='PERT pred.', color='#8E3A71')  # Deep magenta for line
    plt.ylim(-4, 6)
    plt.fill_between(x2, -y2_lower, -y2_upper, color='#8E3A71', alpha=0.1, linewidth=2)  # Deep magenta for band
    plt.legend()
    plt.xlabel('Patients')
    plt.ylabel('Effect')
    plt.title('Plot with Uncertainty Band ' + model_name + ' PERT ' + mode)
    plt.tight_layout()
    plt.show()
    fig2.savefig(model_name + '_PERT_' + mode + '.png')
    plt.close(fig2)
