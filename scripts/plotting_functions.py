# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/7/2025
# Purpose: To plot data from XGBoost

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pingouin import ancova
import seaborn as sns
import statsmodels.stats.multitest as mt
import os
import numpy as np
import pandas as pd
import sys

def plot_xgb_learning_curves(cv_results, metrics, trial_number, version_dir, maximize_metrics):
    """
    Plot learning curves from cross-validation results.
    
    Parameters:
    -----------
    cv_results : pandas DataFrame
        Results from XGBoost cv
    metrics : list
        List of metrics to plot
    trial_number : int
        Current trial number for naming
    version_dir : str
        Directory to save plots
    maximize_metrics : list
        List of metrics that should be maximized
    """
    plt.figure(figsize=(12, 6))
    
    # Plot each metric
    for metric in metrics:
        train_means = cv_results[f'train-{metric}-mean']
        test_means = cv_results[f'test-{metric}-mean']
        train_stds = cv_results[f'train-{metric}-std']
        test_stds = cv_results[f'test-{metric}-std']
        rounds = range(1, len(train_means) + 1)
        
        # Plot means with different line styles
        plt.plot(rounds, train_means, '--', label=f'Training {metric}', alpha=0.8)
        plt.plot(rounds, test_means, '-', label=f'Validation {metric}', alpha=0.8)
        
        # Plot standard deviation bands
        plt.fill_between(rounds, 
                        train_means - train_stds, 
                        train_means + train_stds, 
                        alpha=0.2)
        plt.fill_between(rounds, 
                        test_means - test_stds, 
                        test_means + test_stds, 
                        alpha=0.2)
        
        # Mark best point
        if metric in maximize_metrics:
            best_round = test_means.idxmax() + 1
            best_score = test_means.max()
        else:
            best_round = test_means.idxmin() + 1
            best_score = test_means.min()
            
        plt.scatter(best_round, best_score, color='red', marker='*', s=100,
                   label=f'Best {metric}: {best_score:.4f} (round {best_round})')
    
    plt.title(f'Learning Curves - Trial {trial_number}')
    plt.xlabel('Boosting Round')
    plt.ylabel('Metric Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Ensure directory exists
    os.makedirs(os.path.join(version_dir, 'learning_curves'), exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(version_dir, 'learning_curves', f'learning_curves_trial_{trial_number}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return learning curve data for later analysis
    return {
        'rounds': list(range(1, len(cv_results) + 1)),
        **{f'train-{metric}-mean': list(cv_results[f'train-{metric}-mean']) for metric in metrics},
        **{f'test-{metric}-mean': list(cv_results[f'test-{metric}-mean']) for metric in metrics},
        **{f'train-{metric}-std': list(cv_results[f'train-{metric}-std']) for metric in metrics},
        **{f'test-{metric}-std': list(cv_results[f'test-{metric}-std']) for metric in metrics}
    }

def create_ROC_AUC_precision_recall_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, file_dir, file_name):
    """
    Parameters:
    -----------
        fpr: 
            
        tpr:

        roc_auc:
            ROC AUC value.

        precision:

        recall:

        pr_auc:
            Precision recall AUC value.

        file_dir: String
            Name of the directory to save plots. 
        
        file_name: String
            Name of the file to save plots.


    Function:
    ---------
        - Creates a ROC AUC curve and PR AUC curve on the same file.
        
    Returns:
    --------
        void
    """
    # Create figure
    plt.figure(figsize=(12, 5))

    # --- ROC Curve ---
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # --- Precision–Recall Curve ---
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f"{file_dir}/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_significant_features_box_and_line_plots(mc_cohort, selective_features, responder_group, version_dir):
    """
    Parameters:
    -----------
        mc_cohort: pandas DataFrame
            A dataframe of the mc_cohort.
            
        selective_features: List
            List of features to select for.

        responder_group: String
            Name of responder group column.

        version_dir: String
            Name of directory to save plots to.

    Function:
    ---------
        - Does an ANCOVA test on responder vs non-responder groups of each TFBS in selective_features.
        - Does multiple test correction for all pvalues.
        - Creates a paired box plot for responder groups and a sorted histogram for each patient. 
        
    Returns:
    --------
        void
    """
    features_pvals = {}

    if not selective_features:
        return 

    for tfbs in selective_features:
        # Create Box plot of features for responder groupings
        # Create boxplot for FLI1
        plt.figure(figsize=(12, 10))

        responders = mc_cohort.loc[mc_cohort[responder_group] == 'responder', tfbs]
        non_responders = mc_cohort.loc[mc_cohort[responder_group] == 'non-responder', tfbs]

        # Perform ANCOVA using TFx_C1 as a covariate
        # Create DataFrame with the necessary variables
        ancova_data = mc_cohort[[tfbs, responder_group, 'TFx_C1']].copy()
        
        # Perform ANCOVA
        ancova_results = ancova(data=ancova_data,
                            dv=tfbs,              # Dependent variable (gene expression)
                            between=responder_group,   # Independent variable (response group)
                            covar='TFx_C1',      # Covariate
                            )
        
        pval = ancova_results["p-unc"][0]

        features_pvals[tfbs] = pval

        if (pval <= 0.1):
            print(f"\nANCOVA Results for {tfbs}:")
            print(ancova_results)
            
            # Get adjusted means (controlling for TFx_C1)
            means = ancova_data.groupby(responder_group)[tfbs].mean()
            print("\nAdjusted means (controlling for TFx_C1):")
            print(means)
            

    # Multiple Test Correction
    p_vals = list(features_pvals.values())
    keys = list(features_pvals.keys())
    reject, pvals_corrected, _, _ = mt.multipletests(p_vals, method='fdr_bh')

    # fdr_adjusted_dict = {key: val for key, val in zip(keys, pvals_corrected)}
    fdr_adjusted_dict = dict(zip(keys, pvals_corrected))

    print(f"fdr_adjusted_dict: {fdr_adjusted_dict}\n")

    significant_features = {}

    for key in fdr_adjusted_dict.keys():
        if fdr_adjusted_dict[key] <= 0.05:
            significant_features[key] = fdr_adjusted_dict[key]


    os.makedirs(f"{version_dir}/significant_input_features_plots", exist_ok=True)

    for tfbs in significant_features:

        # Create the boxenplot
        sns.boxplot(data=mc_cohort, 
                    x=responder_group, 
                    y=tfbs, 
                    width=0.4, 
                    palette='Set2'
                    )
        sns.stripplot(x=responder_group, y=tfbs, data=mc_cohort, color='black', size=10, alpha=0.6)

        y_max = mc_cohort[tfbs].max()
        plt.text(0.5, y_max * 1.05, f"p = {pval:.3e}",
                ha='center', va='bottom', fontsize=12)

        # Customize plot
        plt.title(f'{tfbs} Central Depth by Response Group in MC Cohort', fontsize=15)
        plt.xlabel('Response Group', fontsize=15)
        plt.ylabel(f'{tfbs} Expression', fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plt.savefig(f"./feature_plots/paired_box_plot_{tfbs}_mc_cohort.png", dpi=300)
        plt.close()

        mc_cohort_copy = mc_cohort.sort_values(by=tfbs)

        plt.figure(figsize=(10, 12))
        sns.barplot(data=mc_cohort_copy, 
                    x="FLI1", 
                    y=mc_cohort_copy.index, 
                    hue=responder_group,
                    edgecolor='black'
                    )

        plt.title(f"{tfbs} Central Depth by Sample (Colored by Response Group)")
        plt.xlabel(f"{tfbs} Central Depth")
        plt.ylabel("Sample")
        plt.legend(title="Response Group", loc="upper right", frameon=True)
        plt.savefig(f"./feature_plots/{tfbs}_bargraph_mc_cohort.png")
        plt.close()


def TFx_for_each_patient_bargraph(mc_cohort, responder_group, version_dir):
    """
    Parameters:
    -----------
        mc_cohort: pandas DataFrame
            A dataframe of the mc_cohort.

        responder_group: String
            Name of responder group column.

        version_dir: String
            Name of directory to save plots to.

    Function:
    ---------
        - 
        
    Returns:
    --------
        void
    """
    extra_version_dir = version_dir + "/extra-plots"

    os.makedirs(extra_version_dir, exist_ok=True)

    # Peliminary TFx by patient plotting
    mc_cohort_copy = mc_cohort[["TFx_C1", responder_group]].copy()

    mc_cohort_copy = mc_cohort_copy.reset_index().rename(columns={'index': 'patient_id'})
    mc_cohort_copy = mc_cohort_copy.sort_values(by="TFx_C1")

    plt.figure(figsize=(6, 5))
    sns.barplot(data=mc_cohort_copy, x="TFx_C1", y="patient_id", hue=responder_group)
    plt.savefig(f"{extra_version_dir}/TFx_for_each_patient_bargraph.png")
    plt.close()


def create_feature_importance_bargraph(model, version_dir, model_name):
    """
    Parameters:
    -----------
        full_model: XGBoost model
            Model trained on full first cohort.

        version_dir: String
            Name of directory to save plots to.

    Function:
    ---------
        - 
        
    Returns:
    --------
        fi_df: pandas DataFrame
            A df of Feature x Importance Value.
    """
    feature_importance = model.get_score(importance_type='gain')
    print(f"feature_importance: {feature_importance}")

    fi_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values(by="Importance")

    os.makedirs(f"{version_dir}/extra-plots/", exist_ok = True)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=fi_df, x="Feature", y="Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{version_dir}/extra-plots/{model_name}_feature_importance_bargraph.png", dpi=300)
    plt.close()

    return fi_df

def auc_bar_graph_from_auc_pd(auc_df, version_dir, file_name):
    """
    Parameters:
    -----------
        auc_df: pandas DataFrame
            Column(s) of aurocs.

        version_dir: String
            Name of directory to save plots to.

        file_name: String
            Name of file.

    Function:
    ---------
        - 
        
    Returns:
    --------
        void
    """
    # Clears existing details off plot
    plt.clf()

    ax = auc_df.plot.bar(figsize=(23, 18), width=0.75)

    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=16, padding=3)

    plt.xlabel('Feature Selection', fontsize=18)
    plt.xticks(rotation=90, fontsize=16)
    plt.ylabel('AUC', fontsize=18)
    plt.yticks(fontsize=16)
    plt.title('Barplot of AUROCs by Different Feature Selection', fontsize=20)

    # Get current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Create legend with custom labels and larger font
    ax.legend(handles, labels, title='ROC AUC Categories', fontsize=16, title_fontsize=18)

    plt.tight_layout()
    plt.savefig(f"{version_dir}/{file_name}")
    plt.close()

def auc_graph_with_CI(auc_df, version_dir, file_name, conf_int):
    lowers = [conf_int[perm]["CI"][0] for perm in auc_df.index]
    uppers = [conf_int[perm]["CI"][1] for perm in auc_df.index]
    medians = [conf_int[perm]["median"] for perm in auc_df.index]

    lower_errors = auc_df["XGBoost Validation"].values - np.array(lowers)
    upper_errors = np.array(uppers) - auc_df["XGBoost Validation"].values

    plt.clf()

    plt.figure(figsize=(12,10))

    plt.errorbar(x=range(len(auc_df)), y=auc_df["XGBoost Validation"],
                yerr=[lower_errors, upper_errors],
                fmt='none', capsize=5, capthick=2)
    
    plt.scatter(range(len(auc_df)), auc_df["XGBoost Validation"], marker='x', color="red",
                s=100, zorder=5, label="final model AUC")
    
    plt.scatter(range(len(auc_df)), medians, marker='o',
                s=100, zorder=5, label="median AUC from bootstrapping")
    
    for i, (idx, val) in enumerate(zip(auc_df.index, auc_df["XGBoost Validation"])):
        y_pos = val - lower_errors[i] - 0.06
        plt.text(i, y_pos, f'{val:.3f}', ha="center", va="bottom", fontsize=9)

    plt.title("AUROCs by Different Feature Selection with 95% CI and Median Value")
    plt.xticks(range(len(auc_df)), auc_df.index, rotation=90)
    plt.ylabel('AUROC (95% CI)')
    plt.xlabel('Feature Selection')

    plt.legend(loc='lower center', ncol=2)
    
    plt.ylim(0, 1)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(f"{version_dir}/{file_name}")
    plt.close()

def auc_graph_with_CI_v2(auc_dict, version_dir, file_name):
    plt.clf()

    perms = list(auc_dict.keys())
    algos = sorted(auc_dict[perms[0]].keys())
    fsms = list(next(iter(auc_dict.values()))[algos[0]].keys())

    x_labels = []
    x_positions = {}
    x = 0

    for perm in perms:
        for fsm in fsms:
            x_labels.append(f"{perm}\n{fsm}")
            x_positions[(perm, fsm)] = x
            x += 1
        x += 1

    off_set = 0.15
    colors = {algo: f"C{i}" for i, algo in enumerate(algos)}

    plt.figure(figsize=(12,10))

    for i, algo in enumerate(algos):
        xs = []
        ys = []
        yerr_lower = []
        yerr_upper = []
        for perm in perms:
            for fsm in fsms:
                entry = auc_dict[perm][algo][fsm]
                print()
                print(f"entry for {algo} / {fsm}: {entry}")
                median = entry["median"]
                ci_low, ci_high = entry["CI"]

                base_x = x_positions[(perm, fsm)]

                xs.append(base_x + (i - (len(algos) - 1) / 2) * off_set)

                ys.append(median)
                yerr_lower.append(median - ci_low)
                yerr_upper.append(ci_high - median)


        plt.errorbar(
            x=xs, 
            y=ys,
            yerr=[yerr_lower, yerr_upper],
            fmt='o',
            color=colors[algo],
            capsize=5, 
            capthick=2
        )

    plt.title("AUROCs by Different Feature Selection with 95% CI and Median Value")
    plt.xticks(
        [x_positions[(perm, fsm)] for perm in perms for fsm in fsms], 
        [f"{perm}\n{fsm}" for perm in perms for fsm in fsms],
        ha="right",
        rotation=90
    )
    plt.ylabel('AUROC (95% CI)')
    plt.xlabel('Feature Selection')

    legend_handles = []
    for i, algo in enumerate(algos):
        color = colors[algo]
        legend_handles.append(mlines.Line2D([], [], color=color, marker="o", linestyle="none", label=algo))


    plt.legend(handles = legend_handles)
    
    plt.ylim(0, 1)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(f"{version_dir}/{file_name}")
    plt.close()

def auc_graph_with_CI_publication(
    auc_dict, 
    version_dir, 
    file_name,
    title="AUROC by Feature Selection Strategy",
    figsize=(14, 8)
):
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    perms = list(auc_dict.keys())
    algos = sorted(auc_dict[perms[0]].keys())
    fsms = list(next(iter(auc_dict.values()))[algos[0]].keys())

    # ---- X-axis construction ----
    x_labels = []
    x_positions = {}
    x = 0

    for perm in perms:
        for fsm in fsms:
            x_labels.append(f"{perm}\n{fsm}")
            x_positions[(perm, fsm)] = x
            x += 1
        x += 1  # spacing between perm groups

    offset = 0.18
    colors = {algo: f"C{i}" for i, algo in enumerate(algos)}

    # ---- Plot ----
    for i, algo in enumerate(algos):
        xs, ys = [], []
        yerr_lower, yerr_upper = [], []

        for perm in perms:
            for fsm in fsms:
                entry = auc_dict[perm][algo][fsm]
                median = entry["median"]
                ci_low, ci_high = entry["CI"]

                base_x = x_positions[(perm, fsm)]
                x_pos = base_x + (i - (len(algos) - 1) / 2) * offset

                xs.append(x_pos)
                ys.append(median)
                yerr_lower.append(median - ci_low)
                yerr_upper.append(ci_high - median)

                # ---- Label: median ----
                ax.text(
                    x_pos,
                    ci_high + 0.025,
                    f"{median:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=colors[algo]
                )

                # ---- Label: CI ----
                ax.text(
                    x_pos,
                    ci_low - 0.035,
                    f"[{ci_low:.2f}, {ci_high:.2f}]",
                    ha='center',
                    va='top',
                    fontsize=7,
                    color=colors[algo],
                    alpha=0.9
                )

        ax.errorbar(
            x=xs,
            y=ys,
            yerr=[yerr_lower, yerr_upper],
            fmt='o',
            color=colors[algo],
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            markersize=6,
            label=algo
        )

    # ---- Axis formatting ----
    ax.set_xticks([x_positions[(perm, fsm)] for perm in perms for fsm in fsms])
    ax.set_xticklabels(
        [f"{perm}\n{fsm}" for perm in perms for fsm in fsms],
        rotation=45,
        ha="right",
        fontsize=10
    )

    ax.set_ylabel("AUROC (95% CI)", fontsize=12)
    ax.set_xlabel("Feature Selection Strategy", fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')

    ax.set_ylim(0, 1)

    # ---- Grid + spines (publication style) ----
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- Legend ----
    ax.legend(
        frameon=False,
        fontsize=10,
        title="Model",
        title_fontsize=11
    )

    # ---- Layout ----
    plt.tight_layout()
    plt.savefig(f"{version_dir}/{file_name}", dpi=300, bbox_inches='tight')
    plt.close()