import numpy as np
import os
import pymc3 as pm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy import stats
from tabulate import tabulate

# Bounds for the null hypothesis
UPPER_BOUNDS = [0.1, 0.25, 0.5, 1.0]
LOWER_BOUNDS = [-0.1, -0.25, -0.5, -1.0]

# Functions for Bayesian analysis
def compute_bayesian_stats(data: np.ndarray, credibility_level: float =0.95) -> tuple:
    """
    Returns credibility interval and HDI.
    """
    with pm.Model() as model:
        # Standard normal prior
        mu = pm.Normal('mu', mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal('sigma', sigma=1.0)

        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data)

        # Sample from posterior
        trace = pm.sample(5000, return_inferencedata=False)

        # Compute credibility interval
        ci_lower = np.percentile(trace['mu'], 100 * (1 - credibility_level) / 2)
        ci_upper = np.percentile(trace['mu'], 100 * (1 + credibility_level) / 2)

        # Implement HDI
        hdi = pm.hdi(trace['mu'], hdi_prob=credibility_level)

        # hdi output is a dictionary with keys 'lower' and 'upper'
        hdi_lower, hdi_upper = hdi[0], hdi[1]
    
    return (ci_lower, ci_upper), (hdi_lower, hdi_upper)

def main():
    parser = ArgumentParser(
        prog='statistical_tests',
        description='Loads .csv files and and runs some tests.',
        epilog='Only for use on experimental data.'
    )
    parser.add_argument('perm_data', 
                        help='The permutation data to be tested.',
                        type=str)
    parser.add_argument('null_data',
                        help='The null data to be tested.',
                        type=str)
    parser.add_argument('wasserstein', 
                        help='The Wasserstein data.',
                        type=str)
    
    args = parser.parse_args()
    fn_perm = args.perm_data
    fn_null = args.null_data
    fn_wasser = args.wasserstein

    # Load the data
    print('Loading the data.')
    perm_data = np.loadtxt(fn_perm, delimiter=',', dtype=float)
    null_data = np.loadtxt(fn_null, delimiter=',', dtype=float)
    wasser_data = np.loadtxt(fn_wasser, delimiter=',', dtype=float)

    # Set the critical threshold
    alpha = 0.05

    # Set the sample size
    n = perm_data.shape[0]

    # Summary statistics
    perm_mean = np.mean(perm_data)
    null_mean = np.mean(null_data)
    perm_std = np.std(perm_data)
    null_std = np.std(null_data)
    _, perm_minmax, _, perm_var, perm_skew, perm_ker = stats.describe(perm_data)
    _, null_minmax, _, null_var, null_skew, null_ker = stats.describe(null_data)
    
    # Save histograms of the mean data
    print('Generating histograms, saving figures, and summary statistics.')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,5))
    axs[0].hist(perm_data, bins=100)
    axs[0].set_title('Intervention Means')
    axs[1].hist(null_data, bins=100)
    axs[1].set_title('Null Means')
    fig.tight_layout()
    plt.savefig('histograms_data.png')

    # Save the Wasserstein Plots
    print('Generating Wasserstein plots and saving.')
    fig, axs = plt.subplots(figsize=(7,5))
    cax = axs.imshow(wasser_data, cmap='viridis')
    fig.colorbar(cax)
    labels = ['Intervention Distribution', 'Model Distribution', 'Random Sample']
    axs.set_title('Wasserstein Distances')
    axs.set_xlabel('Distributions')
    axs.set_ylabel('Distributions') 
    axs.set_xticks(np.arange(len(labels)))
    axs.set_yticks(np.arange(len(labels)))
    axs.set_xticklabels(labels)
    axs.set_yticklabels(labels)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig('wasserstein_data.png')

    # Save summary statistics
    perm_summary = ['Intervention Data', perm_mean, perm_minmax, perm_var, perm_skew, perm_ker]
    null_summary = ['Null Data', null_mean, null_minmax, null_var, null_skew, null_ker]
    headers = ['Dataset',
               'Mean',
               '(Min, Max)',
               'Variance',
               'Skewness',
               'Kurtosis']
    s = ('Summary Statistics\n\n' + tabulate([perm_summary, null_summary], headers=headers) + '\n\n')
    fn = 'statistic_results.txt'
    with open(fn, "a") as f:
            f.write(s)
    
    # Loop through and do the tests
    print('Conducting the tests.')

    # First test that the perm and null have identical means
    t_statistic, p_value = stats.ttest_ind(perm_data, null_data, equal_var=False) # The variances will be different
    difference_test = "Reject" if p_value / 2 < alpha else "Fail"
    s = ('Difference in means between Intervention and null data:\n' + 
         f'T-statistic: {t_statistic}\n' +
         f'p-values: {p_value}\n' + 
         f'Reject/Fail: {difference_test}\n\n')
    with open(fn, "a") as f:
            f.write(s)

    for upper_b, lower_b in zip(UPPER_BOUNDS, LOWER_BOUNDS):
        # Permutation tests
        # Equivalence test
        t_statistic_low, p_value_low = stats.ttest_1samp(a=perm_data, popmean=lower_b, alternative="greater")
        lower_bound_test = p_value_low / 2 < alpha
        t_statistic_high, p_value_high = stats.ttest_1samp(a=perm_data, popmean=upper_b, alternative="less")
        upper_bound_test = p_value_high / 2 < alpha
        equivalence_test = "Reject" if (lower_bound_test and upper_bound_test) else "Fail"


        # Confidence interval 0.95
        conf_int = stats.norm.interval(0.95, loc=perm_mean, scale=perm_std/np.sqrt(n))

        # Credibility interval 0.95
        cred_int, hdi_int = compute_bayesian_stats(perm_data, 0.95)

        # Store the results
        perm_stats = ['Permutation Statistics', 
                       t_statistic_low, 
                       p_value_low, 
                       t_statistic_high,
                       p_value_high,
                       equivalence_test,
                       conf_int,
                       cred_int,
                       hdi_int]
        
        # Null tests
        t_statistic_low, p_value_low = stats.ttest_1samp(a=null_data, popmean=lower_b, alternative="greater")
        lower_bound_test = p_value_low / 2 < alpha
        t_statistic_high, p_value_high = stats.ttest_1samp(a=null_data, popmean=upper_b, alternative="less")
        upper_bound_test = p_value_high / 2 < alpha
        equivalence_test = "Reject" if (lower_bound_test and upper_bound_test) else "Fail"


        # Confidence interval 0.95
        conf_int = stats.norm.interval(0.95, loc=null_mean, scale=null_std/np.sqrt(n))

        # Credibility interval 0.95
        cred_int, hdi_int = compute_bayesian_stats(null_data, 0.95)

        null_stats = ['Null Statistics', 
                       t_statistic_low, 
                       p_value_low, 
                       t_statistic_high,
                       p_value_high,
                       equivalence_test,
                       conf_int,
                       cred_int,
                       hdi_int]

        # Print the results
        headers = ['Dataset',
                    'T Statistic Lower', 
                   'P Value Lower', 
                   'T Statistic Higher', 
                   'P Value Higher', 
                   'Equivalence Test',
                   'Confidence Interval',
                   'Credibility Interval',
                   'HDI Interval']
        
        s = (f'The statistical results for null hypothesis bound {lower_b} and {upper_b}:\n\n' +  
             tabulate([perm_stats, null_stats], headers=headers) + '\n\n')
        print(s)

        # Save the results
        print('Saving results.')
        with open(fn, "a") as f:
            f.write(s)

if __name__ == "__main__":
    main()