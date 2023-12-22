
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm
import statsmodels.stats.api as sms
import scipy.stats as stats
import math
import os
from docx import Document
from docx.shared import Inches
import yaml

config_path = 'config.yaml'

class ABTestAnalyzer:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.test_path = config['test_path']
        self.conversion_metric = config['conversion_metric']
        self.id_column = config['id_column']
        self.spend_column = config['spend_column']
        self.binary = config['binary']
        self.MDE = config['MDE']
        self.significance_level = config['significance_level']
        self.power = config['power']
        self.group_ratio = config['group_ratio']
        self.SRM_alpha = config['SRM_alpha']
        self.AB_metric= config['AB_metric']
        self.AB_alpha= config['AB_alpha']
        self.NE_alpha= config['NE_alpha']
        self.test_data = None

    def load_data(self):
        try:
            self.test_data = pd.read_csv(self.test_path, low_memory=False)
            self.test_data['date'] = pd.to_datetime(self.test_data['date'], format='%m/%d/%y')
        except Exception as e:
            print(f"Error loading data: {e}")
    
    # chi-square test
    def chi_square(self, data):
        try:
            AB_test_data = data[data['experiment'] == self.AB_metric]
            observed = AB_test_data.groupby('group')['experiment'].count().values
            expected = [AB_test_data.shape[0]*0.5]*2
            
            # perform Chi-Square Goodness of Fit Test
            chi_stats, pvalue = stats.chisquare(f_obs=observed, f_exp=expected)
            return {
                'pvalue': pvalue,
                'chi_stats': chi_stats,
                'observed': observed,
                'expected': expected
            }    
        except Exception as e:
            print(f"Error in chi_square: {e}")

    def chi_square_validate(self, data):
        try: 
            stats= self.chi_square(data)
            if stats['pvalue'] <  self.SRM_alpha:
                return 'Reject H0 and conclude that there is statistical significance in the ratio of samples not being 1:1. Therefore, there is SRM.'
            else:
                return 'Fail to reject H0. No mismatch in the distribution of samples between the control and treatment groups.'
        except Exception as e:
            print(f"Error in chi_square_validate: {e}")

    # AB test
    def AB_test(self, data):
        try:
            AB_test_data = data[data['experiment'] == self.AB_metric]
            control = AB_test_data[AB_test_data['group'] == 0][self.conversion_metric]
            treatment = AB_test_data[AB_test_data['group'] == 1][self.conversion_metric]
            # Get stats
            AB_control_sum = control.sum()          # Control Sum
            AB_treatment_sum = treatment.sum()      # Treatment Sum
            AB_control_mean = control.mean()        # Control Mean
            AB_treatment_mean = treatment.mean()    # Treatment Mean
            AB_control_size = control.count()       # Control Sample Size
            AB_treatment_size = treatment.count()   # Treatment Sample Size

            # Create two descriptive statistics objects using test and control data
            desc_stats_test = sm.stats.DescrStatsW(treatment)
            desc_stats_control = sm.stats.DescrStatsW(control)
            # Compare the means of the two datasets
            cm = sms.CompareMeans(desc_stats_test, desc_stats_control)
            # Calculate the confidence interval for the difference between the means (using unequal variances)
            lb, ub = cm.tconfint_diff(usevar='unequal')
            # Calculate lift between test and control
            lower_lift = lb / AB_control_mean
            upper_lift = ub / AB_control_mean
            # lift
            absolute_lift = AB_treatment_mean - AB_control_mean
            relative_lift = (AB_treatment_mean - AB_control_mean) / AB_control_mean
            # perform AB test
            AB_tstat, AB_pvalue, AB_df = ttest_ind(control, treatment)
            return {
                'pvalue': AB_pvalue,
                'control_mean': AB_control_mean,
                'treatment_mean': AB_treatment_mean,
                'control_size': AB_control_size,
                'treatment_size': AB_treatment_size,
                'lower_lift': lower_lift,
                'upper_lift': upper_lift,
                'lb': lb,
                'ub': ub,
                'absolute_lift': absolute_lift,
                'relative_lift': relative_lift
            }
        except Exception as e:
            print(f"Error in AB_test: {e}")
    
    def AB_validate(self, data):
        try: 
            stats= self.AB_test(data)
            if stats['pvalue'] <  self.AB_alpha:
                return 'Reject H0 and conclude that there is statistical significance in the difference of conversion between two variables.'
            else:
                return 'Fail to reject H0.'
        except Exception as e:
            print(f"Error in AB_validate: {e}")
 
    def novelty_validate(self, data):
        try: 
            # average sales per user per day
            AB_test_data = data[data['experiment'] == self.AB_metric]
            AB_per_day = AB_test_data.groupby(['group','date'])[self.conversion_metric].mean()
            AB_ctrl_conversion = AB_per_day.loc[0]
            AB_trt_conversion = AB_per_day.loc[1]
            # Get the day range of experiment
            exp_days = range(1, AB_test_data['date'].nunique() + 1)
            # Preparing data for regression
            combined_data = pd.DataFrame({
                'date': AB_ctrl_conversion.index,
                'AB_ctrl_conversion': AB_ctrl_conversion.values,
                'AB_trt_conversion': AB_trt_conversion.values
            })

            # Adding a time index (number of days since start of experiment)
            combined_data['time_index'] = (combined_data['date'] - combined_data['date'].min()).dt.days

            # Setting up the regression model for Group 1 (treatment group)
            X = sm.add_constant(combined_data['time_index'])  # Adding a constant to the model
            y = combined_data['AB_trt_conversion']

            # Fit the linear regression model
            model = sm.OLS(y, X).fit()

            # Extracting the required information from the model
            return {
                'pvalue': model.pvalues['time_index'],  # p-value for the time index
                'rsquared': model.rsquared,  # R-squared value of the model
                'coef': model.params['time_index'],  # Coefficient for the time index
                'exp_days': exp_days,
                'AB_ctrl_conversion': AB_ctrl_conversion,
                'AB_trt_conversion': AB_trt_conversion
            }
        except Exception as e:
            print(f"Error in novelty_validate: {e}")
           
    def plot_novelty_effect(self, data):
        try: 
            stats=self.novelty_validate(data)
            f, ax = plt.subplots(figsize=(12, 6))
            # Generate plots
            ax.plot(stats['exp_days'], stats['AB_ctrl_conversion'], label='Control', color='b', marker='o')
            ax.plot(stats['exp_days'], stats['AB_trt_conversion'], label='Treatment', color='r', marker='o')

            # # Format plot
            ax.set_xticks(stats['exp_days'])
            plt.title('Daily Conversion Rates by Group')
            plt.ylabel('Conversion Rate per Day')
            ax.set_xlabel('Days in the Experiment')
            ax.legend()
            return plt 
        except Exception as e:
            print(f"Error in plot_novelty_effect: {e}")


# Example usage
analyzer=ABTestAnalyzer(config_path)
analyzer.load_data()
# Call the functions and store the results
observed = analyzer.chi_square(analyzer.test_data)['observed']
expected = analyzer.chi_square(analyzer.test_data)['expected']
chisquare_pvalue=analyzer.chi_square(analyzer.test_data)['pvalue']
AB_pvalue=analyzer.AB_test(analyzer.test_data)['pvalue']
avg_control_conversion=analyzer.AB_test(analyzer.test_data)['control_mean']
avg_treatment_conversion=analyzer.AB_test(analyzer.test_data)['treatment_mean']
AB_control_size=analyzer.AB_test(analyzer.test_data)['control_size']
AB_treatment_size=analyzer.AB_test(analyzer.test_data)['treatment_size']
absolute_lift=analyzer.AB_test(analyzer.test_data)['absolute_lift']
relative_lift=analyzer.AB_test(analyzer.test_data)['relative_lift']
lower=analyzer.AB_test(analyzer.test_data)['lb']
upper=analyzer.AB_test(analyzer.test_data)['ub']
lower_lift=analyzer.AB_test(analyzer.test_data)['lower_lift']
upper_lift=analyzer.AB_test(analyzer.test_data)['upper_lift']
validate_SR = analyzer.chi_square_validate(analyzer.test_data)
validate_AB = analyzer.AB_validate(analyzer.test_data)
novelty_pvalue = analyzer.novelty_validate(analyzer.test_data)['pvalue']
novelty_plot=analyzer.plot_novelty_effect(analyzer.test_data)
start_date=analyzer.test_data['date'].min().date()
end_date=analyzer.test_data['date'].max().date()
sample_duration=(analyzer.test_data['date'].max() - analyzer.test_data['date'].min()).days+1
test_data_columns=analyzer.test_data.columns.values

# Define the directory
directory = 'output'
# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the results to a txt file
with open(os.path.join(directory, 'AB_post-test.txt'), 'w') as f:
    f.write(f'# AB Test Post-test Results \n')
    f.write(f'\n## Test Parameters \n')
    f.write(f'Minimum Detectable Effect: {analyzer.MDE}\n')
    f.write(f'AB Test significance Level: {analyzer.AB_alpha}\n')
    f.write(f'Power: {analyzer.power}\n')
    f.write(f'\n## Test Data \n')
    f.write(f'Number of rows: {analyzer.test_data.shape[0]}, Number of columns: {analyzer.test_data.shape[1]}\n')
    f.write(f'Test data columns: {test_data_columns}\n')
    f.write(f'Conversion metric: {analyzer.conversion_metric}\n')
    f.write(f'Start date: {start_date}, End date: {end_date}, Duration: {sample_duration} days\n')
    f.write(f'Observed: {observed}, Expected: {expected}\n')
    f.write(f'Average control conversion: {avg_control_conversion:.4f}, Average treatment conversion: {avg_treatment_conversion:4f}\n')
    f.write(f'\n## Validity Test - SRM Test (Sample Ratio Mismatch)\n')
    f.write(f'H0: The ratio of samples is 1:1.\n')
    f.write(f'H1: The ratio of samples is not 1:1.\n')
    f.write(f'SRM significance level: {analyzer.SRM_alpha}, SRM p-value: {chisquare_pvalue:.3f}\n')
    f.write(f'SRM validation: {validate_SR}\n')
    f.write(f'\n## Validity Test - Novelty Effect Test \n')
    f.write(f'H0: There is no novelty effect.\n')
    f.write(f'H1: There is novelty effect.\n')
    f.write(f'Novelty significance level: {analyzer.NE_alpha}, Novelty p-value: {novelty_pvalue:.3f}\n')
    if novelty_pvalue < analyzer.NE_alpha:
        f.write('Novelty Effect Test Validation: Reject H0 and conclude that there is novelty effect. There is no novelty effect. It suggests that the observed changes in the conversion rate are statistically significant and can be attributed to the novelty of the treatment (e.g., a new feature or interface).\n')
    else:
        f.write('Novelty Effect Test Validation: Fail to reject H0. There is no novelty effect. In other words, it asserts that any observed changes in the conversion rate (or other metrics of interest) over time are due to random chance rather than a genuine novelty effect. \n')
    f.write(f'\n## AB Test Results \n')
    f.write(f'H0: There is no difference in conversion between the control and treatment groups.\n')
    f.write(f'H1: There is difference in conversion between the control and treatment groups.\n')
    f.write(f'AB test significance level: {analyzer.AB_alpha}, AB test p-value: {AB_pvalue:.3f}\n')
    f.write(f'AB test validation: {validate_AB}\n')
    f.write(f'\nAbsolute difference: {absolute_lift:.4f}\n')
    f.write(f'Relative difference: {relative_lift*100:.2f}%\n')
    f.write(f'Absolute difference CI: {lower:.4f} to {upper:.4f}\n')
    f.write(f'Relative difference CI: {lower_lift*100:.2f}% to {upper_lift*100:.2f}%\n')
    f.write(f'\n## AB Test Conclusion \n')
    if chisquare_pvalue < analyzer.SRM_alpha:
        f.write('Reject H0 and conclude that there is statistical significance in the ratio of samples not being 1:1. It indicates a potential issue in the experimental setup, such as a mismatch in the distribution of samples between the control and treatment groups.\n')
    else:
        if AB_pvalue < analyzer.AB_alpha:
            f.write(f'During the test, we noticed a {relative_lift*100:.2f}% boost in performance compared to the control group. This outcome was statistically significant, with a {int(100*(1-analyzer.AB_alpha))}% confidence interval ranging from {lower_lift*100:.2f}% to {upper_lift*100:.2f}%.')
            if lower_lift < analyzer.MDE:
                f.write(f'\nHowever, the lower bound of confidence interval {lower_lift*100:.2f}% is smaller than the minimum detectable effect {int(100*analyzer.MDE)}%. Therefore, we cannot conclude that the test is successful. We need to repeat the test with a larger sample size to ascertain whether the new test genuinely results in a {int(100*analyzer.MDE)}% increase.')
            else:
                f.write(f'\nThe lower bound of confidence interval {lower_lift*100:.2f}% is greater than the minimum detectable effect {int(100*(1-analyzer.MDE))}%. Therefore, we can conclude that the test is successful. We can roll out the new test to all users.')
        else:
            f.write(f'The test did not result in a statistically significant difference in performance compared to the control group. Therefore, we cannot conclude that the test is successful. We need to repeat the test with a larger sample size to ascertain whether the new test genuinely results in a {int(100*analyzer.MDE)}% increase.')
    #Save the plot in the directory
    plt.savefig(os.path.join(directory, 'AB_test.png'))
