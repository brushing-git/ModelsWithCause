#!/usr/bin/env python3
"""
Generate publication-quality confidence interval plots for ML paper.
Follows best practices for ML journal and conference submissions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Define color palette (colorblind-friendly)
COLORS = {
    'nade': '#E69F00',       # Orange
    'ed_tr': '#56B4E9',      # Sky blue  
    'd_tr': '#009E73',       # Bluish green
    'moe': '#CC79A7',        # Reddish purple
}

MARKERS = {
    'orig': 'o',   # Circle for original ordering
    'rand': 's',   # Square for random ordering
}

# ============================================================================
# DATA: 0.95 Confidence Intervals
# ============================================================================

# Trained Models
data_trained = {
    'NADE': {
        'perm': (-0.051285, 0.058017),
        'rand': (-0.083083, 0.064246),
        'var_perm': (1.07764356973127, 1.155067013044909),
        'var_rand': (1.4525576105080145, 1.5569168021515063),
    },
    'ED-Tr.': {
        'perm': (-0.090039, -0.038207),
        'rand': (-0.246838, 0.399204),
        'var_perm': (0.5110259189096568, 0.5477406429388351),
        'var_rand': (6.369499164355783, 6.827116665484422),
        'mean_diff': (13.670478588617941, 13.81940760700313),
        'rand_diff': (13.344371121947876, 13.99314922760341),
    },
    'D-Tr.': {
        'perm': (-0.15757, -0.090131),
        'rand': (-0.260696, 0.412721),
        'var_perm': (0.6648976580331973, 0.7126673172990544),
        'var_rand': (6.639404017040572, 7.116412867636576),
        'mean_diff': (13.46670288205913, 13.606253407338274),
        'rand_diff': (13.129136620748493, 13.791795573774436),
    },
    'MOE': {
        'perm': (0.002411, 0.026108),
        'rand': (-0.226417, 0.42417),
        'var_perm': (0.23363001106812223, 0.25041518979174077),
        'var_rand': (6.414320734680001, 6.875158447395011),
        'mean_diff': (13.881601163370902, 14.035980608387781),
        'rand_diff': (13.537264420846562, 14.182564212989067),
    },
}

# No Trained Models  
data_notrained = {
    'NADE': {
        'perm': (-0.070613, -0.020076),
        'rand': (-0.034386, 0.014751),
        'var_perm': (0.498255454994419, 0.5340526833721576),
        'var_rand': (0.4844594740593255, 0.5192655283811092),
    },
    'ED-Tr.': {
        'perm': (0.957825, 1.12261),
        'rand': (-0.306221, 0.25644),
        'var_perm': (1.6246599874537464, 1.741383896894351),
        'var_rand': (5.547424749405773, 5.945980206595946),
    },
    'D-Tr.': {
        'perm': (-0.358033, -0.126329),
        'rand': (-0.418674, 0.262643),
        'var_perm': (2.28443547529568, 2.44856104101434),
        'var_rand': (6.717284038589112, 7.1998881895265185),
    },
    'MOE': {
        'perm': (-0.20991, 0.077225),
        'rand': (-0.369353, 0.506996),
        'var_perm': (2.830943849159151, 3.0343333805272317),
        'var_rand': (8.640171493310659, 9.260925745107727),
    },
}


def ci_to_point_and_error(ci):
    """Convert confidence interval tuple to point estimate and error bars."""
    lower, upper = ci
    point = (lower + upper) / 2
    error = (upper - lower) / 2
    return point, error


def plot_mean_permutations(data, title_suffix, filename, include_nade=True):
    """
    Plot mean log probability differences for permutation tests.
    Equivalence thresholds: (-0.1, 0.1) inner, (-0.5, 0.5) outer
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    models = ['NADE', 'ED-Tr.', 'D-Tr.', 'MOE'] if include_nade else ['ED-Tr.', 'D-Tr.', 'MOE']
    color_keys = ['nade', 'ed_tr', 'd_tr', 'moe'] if include_nade else ['ed_tr', 'd_tr', 'moe']
    
    x_positions = np.arange(len(models))
    width = 0.35
    
    for i, (model, ckey) in enumerate(zip(models, color_keys)):
        # Original ordering
        point_perm, err_perm = ci_to_point_and_error(data[model]['perm'])
        ax.errorbar(x_positions[i] - width/2, point_perm, yerr=err_perm,
                   fmt=MARKERS['orig'], color=COLORS[ckey], capsize=4, capthick=1.5,
                   markersize=8, markeredgewidth=1.5, markeredgecolor='white',
                   label=f'{model} Orig.' if i == 0 else None, zorder=3)
        
        # Random ordering
        point_rand, err_rand = ci_to_point_and_error(data[model]['rand'])
        ax.errorbar(x_positions[i] + width/2, point_rand, yerr=err_rand,
                   fmt=MARKERS['rand'], color=COLORS[ckey], capsize=4, capthick=1.5,
                   markersize=7, markeredgewidth=1.5, markeredgecolor='white',
                   label=f'{model} Rand.' if i == 0 else None, zorder=3)
    
    # Add equivalence threshold bands
    ax.axhspan(-0.1, 0.1, alpha=0.15, color='gray', label='±0.1 threshold', zorder=1)
    ax.axhspan(-0.5, -0.1, alpha=0.08, color='gray', zorder=1)
    ax.axhspan(0.1, 0.5, alpha=0.08, color='gray', zorder=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, zorder=2)
    ax.axhline(y=0.1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
    ax.axhline(y=-0.1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7, zorder=2)
    ax.axhline(y=-0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7, zorder=2)
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models)
    ax.set_ylabel('Mean Difference in Log Probabilities', fontweight='bold')
    ax.set_title(f'95% CI: Mean Log Probability Differences\n{title_suffix}', fontweight='bold')
    
    # Custom legend
    legend_elements = []
    for model, ckey in zip(models, color_keys):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[ckey],
                                      markersize=8, label=f'{model} Orig.', markeredgewidth=1, markeredgecolor='white'))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS[ckey],
                                      markersize=7, label=f'{model} Rand.', markeredgewidth=1, markeredgecolor='white'))
    
    # Add threshold indicators to legend
    legend_elements.append(mpatches.Patch(facecolor='gray', alpha=0.15, label='±0.1 threshold'))
    legend_elements.append(Line2D([0], [0], color='gray', linestyle=':', linewidth=0.8, label='±0.5 threshold'))
    
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8, 
              framealpha=0.95, edgecolor='0.8')
    
    # Set y-axis limits with some padding
    all_vals = []
    for model in models:
        all_vals.extend(data[model]['perm'])
        all_vals.extend(data[model]['rand'])
    y_min, y_max = min(all_vals), max(all_vals)
    padding_bottom = (y_max - y_min) * 0.15
    padding_top = (y_max - y_min) * 0.75  # Extra space at top for legend
    ax.set_ylim(min(-0.6, y_min - padding_bottom), max(0.8, y_max + padding_top))
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_variance_permutations(data, title_suffix, filename, include_nade=True):
    """
    Plot variance log probability differences for permutation tests.
    Null hypothesis threshold: 1.0
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    models = ['NADE', 'ED-Tr.', 'D-Tr.', 'MOE'] if include_nade else ['ED-Tr.', 'D-Tr.', 'MOE']
    color_keys = ['nade', 'ed_tr', 'd_tr', 'moe'] if include_nade else ['ed_tr', 'd_tr', 'moe']
    
    x_positions = np.arange(len(models))
    width = 0.35
    
    for i, (model, ckey) in enumerate(zip(models, color_keys)):
        # Original ordering
        point_perm, err_perm = ci_to_point_and_error(data[model]['var_perm'])
        ax.errorbar(x_positions[i] - width/2, point_perm, yerr=err_perm,
                   fmt=MARKERS['orig'], color=COLORS[ckey], capsize=4, capthick=1.5,
                   markersize=8, markeredgewidth=1.5, markeredgecolor='white', zorder=3)
        
        # Random ordering
        point_rand, err_rand = ci_to_point_and_error(data[model]['var_rand'])
        ax.errorbar(x_positions[i] + width/2, point_rand, yerr=err_rand,
                   fmt=MARKERS['rand'], color=COLORS[ckey], capsize=4, capthick=1.5,
                   markersize=7, markeredgewidth=1.5, markeredgecolor='white', zorder=3)
    
    # Add null hypothesis threshold
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.8, 
               label='Null threshold (1.0)', zorder=2)
    ax.axhspan(0.8, 1.2, alpha=0.1, color='red', zorder=1)
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models)
    ax.set_ylabel('Variance of Log Probability Differences', fontweight='bold')
    ax.set_title(f'95% CI: Variance of Log Probability Differences\n{title_suffix}', fontweight='bold')
    
    # Custom legend
    legend_elements = []
    for model, ckey in zip(models, color_keys):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[ckey],
                                      markersize=8, label=f'{model} Orig.', markeredgewidth=1, markeredgecolor='white'))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS[ckey],
                                      markersize=7, label=f'{model} Rand.', markeredgewidth=1, markeredgecolor='white'))
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.2, label='Null (1.0)'))
    
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8,
              framealpha=0.95, edgecolor='0.8')
    
    # Set y-axis limits
    all_vals = []
    for model in models:
        all_vals.extend(data[model]['var_perm'])
        all_vals.extend(data[model]['var_rand'])
    y_min, y_max = min(all_vals), max(all_vals)
    padding_bottom = (y_max - y_min) * 0.15
    padding_top = (y_max - y_min) * 0.55  # Extra space at top for legend
    ax.set_ylim(min(0.0, y_min - padding_bottom), max(0.8, y_max + padding_top))
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_mean_difference_trained(data, filename):
    """
    Plot mean difference in log probabilities for trained models.
    Only ED-Tr., D-Tr., and MOE have mean_diff/rand_diff values.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    models = ['ED-Tr.', 'D-Tr.', 'MOE']
    color_keys = ['ed_tr', 'd_tr', 'moe']
    
    x_positions = np.arange(len(models))
    width = 0.35
    
    for i, (model, ckey) in enumerate(zip(models, color_keys)):
        # Mean difference (original)
        point_diff, err_diff = ci_to_point_and_error(data[model]['mean_diff'])
        ax.errorbar(x_positions[i] - width/2, point_diff, yerr=err_diff,
                   fmt=MARKERS['orig'], color=COLORS[ckey], capsize=4, capthick=1.5,
                   markersize=8, markeredgewidth=1.5, markeredgecolor='white', zorder=3)
        
        # Random difference
        point_rand, err_rand = ci_to_point_and_error(data[model]['rand_diff'])
        ax.errorbar(x_positions[i] + width/2, point_rand, yerr=err_rand,
                   fmt=MARKERS['rand'], color=COLORS[ckey], capsize=4, capthick=1.5,
                   markersize=7, markeredgewidth=1.5, markeredgecolor='white', zorder=3)
    
    # Add reference lines at 0 with threshold bands
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=2)
    ax.axhspan(-0.1, 0.1, alpha=0.1, color='gray', zorder=1)
    ax.axhspan(-0.5, 0.5, alpha=0.05, color='gray', zorder=1)
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models)
    ax.set_ylabel('Mean Difference in Log Probabilities', fontweight='bold')
    ax.set_title('95% CI: Mean Log Probability Differences\n(Trained Models)', fontweight='bold')
    
    # Custom legend
    legend_elements = []
    for model, ckey in zip(models, color_keys):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[ckey],
                                      markersize=8, label=f'{model} Orig.', markeredgewidth=1, markeredgecolor='white'))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS[ckey],
                                      markersize=7, label=f'{model} Rand.', markeredgewidth=1, markeredgecolor='white'))
    
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8,
              framealpha=0.95, edgecolor='0.8')
    
    # Set y-axis limits with some padding
    all_vals = []
    for model in models:
        all_vals.extend(data[model]['mean_diff'])
        all_vals.extend(data[model]['rand_diff'])
    y_min, y_max = min(all_vals), max(all_vals)
    padding_bottom = (y_max - y_min) * 0.1
    padding_top = (y_max - y_min) * 0.35  # Extra space at top for legend
    ax.set_ylim(y_min - padding_bottom, y_max + padding_top)
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {filename}")


if __name__ == '__main__':
    output_dir = '/Users/brucerushing/Documents/PythonPrograms/CausalReasoning/v2/figures/'
    
    # 1. Mean permutations - Trained models
    plot_mean_permutations(
        data_trained, 
        'Permutation Tests (Trained Models)',
        f'{output_dir}mean-perms-train.pdf',
        include_nade=True
    )
    
    # 2. Mean permutations - No Trained models  
    plot_mean_permutations(
        data_notrained,
        'Permutation Tests (Untrained Models)',
        f'{output_dir}mean-perms-notrain.pdf',
        include_nade=True
    )
    
    # 3. Mean difference - Trained models
    plot_mean_difference_trained(
        data_trained,
        f'{output_dir}mean-dist-train.pdf'
    )
    
    # 4. Variance permutations - Trained models
    plot_variance_permutations(
        data_trained,
        'Permutation Tests (Trained Models)',
        f'{output_dir}var-perms-train.pdf',
        include_nade=True
    )
    
    # 5. Variance permutations - No Trained models
    plot_variance_permutations(
        data_notrained,
        'Permutation Tests (Untrained Models)',
        f'{output_dir}var-perms-notrain.pdf',
        include_nade=True
    )
    
    print("\nAll figures generated successfully!")
