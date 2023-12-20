import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm
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

        self.pretest_path = config['pretest_path']
        self.test_path = config['test_path']
        self.conversion_metric = config['conversion_metric']
        self.id_column = config['id_column']
        self.spend_column = config['spend_column']
        self.binary = config['binary']
        self.MDE = config['MDE']
        self.significance_level = config['significance_level']
        self.power = config['power']
        self.group_ratio = config['group_ratio']
        self.AA_alpha = config['AA_alpha']
        self.pretest_data = None
        self.test_data = None

    def load_data(self):
        try:
            self.pretest_data = pd.read_csv(self.pretest_path, low_memory=False)
            self.test_data = pd.read_csv(self.test_path, low_memory=False)
            self.pretest_data['date'] = pd.to_datetime(self.pretest_data['date'], format='%m/%d/%y')
            self.test_data['date'] = pd.to_datetime(self.test_data['date'], format='%m/%d/%y')
        except Exception as e:
            print(f"Error loading data: {e}")

    def calculate_effect_size(self, data):
        binary = self.binary
        try:
            data[self.conversion_metric] = data[self.conversion_metric].astype(float)
            if binary==True:
                avg_conversion = data[self.conversion_metric].mean()
                effect_size = sm.stats.proportion_effectsize(avg_conversion, avg_conversion * (1 + self.MDE))
            else:
                average_conversion = data[self.conversion_metric].mean()
                std_metric = data[self.conversion_metric].std()
                effect_size = average_conversion * self.MDE / std_metric
            return effect_size
        except Exception as e:
            print(f"Error calculating effect size: {e}")

    def calculate_sample_size(self, data):
        binary = self.binary
        try: 
            effect_size = self.calculate_effect_size(data)
            sample_size = tt_ind_solve_power(effect_size=effect_size, alpha=self.significance_level, power=self.power, ratio=self.group_ratio)
            return sample_size
        except Exception as e:
            print(f"Error calculating sample size: {e}")

    def test_duration(self, data):
        try: 
            sample_size = self.calculate_sample_size(data)
            daily_unique_id = data.groupby('date')[self.id_column].nunique()
            start_date = daily_unique_id.index.min()
            end_date = daily_unique_id.index.max()
            sample_duration = (end_date - start_date).days + 1
            avg_daily_unique = daily_unique_id.mean()
            test_duration = sample_size * (1 + self.group_ratio) / avg_daily_unique
            adjusted_test_duration = math.ceil(test_duration / 7) * 7
            return adjusted_test_duration, start_date, end_date, sample_duration
        except Exception as e:
            print(f"Error calculating test duration: {e}")
    
    def budget(self, data):
        sample_size = self.calculate_sample_size(data)
        spend = data[self.spend_column].sum()
        count = data[self.id_column].nunique()
        avg_cpm = spend / count
        budget = avg_cpm * sample_size * (1 + self.group_ratio)
        return budget

    def AA_duration(self, data):
        AA_test_data = data[data['experiment'] == 'AA_test']
        try:
            start_date = AA_test_data['date'].min()
            end_date = AA_test_data['date'].max()
            duration = (end_date - start_date).days + 1
            return start_date, end_date, duration
        except Exception as e:
            print(f"Error calculating AA duration: {e}")
            

    def AA_test(self, data):
        AA_test_data = data[data['experiment'] == 'AA_test']
        control_group = AA_test_data[AA_test_data['group'] == 0][self.conversion_metric]
        treatment_group = AA_test_data[AA_test_data['group'] == 1][self.conversion_metric]
        try: 
            if data[self.conversion_metric].nunique() == 2:
                _, p_value, _ = proportions_chisquare([control_group.sum(), treatment_group.sum()], nobs=[control_group.count(), treatment_group.count()])
            else:
                _, p_value, _ = ttest_ind(control_group, treatment_group, usevar='unequal')

            return p_value, control_group.mean(), treatment_group.mean()
        except Exception as e:
            print(f"Error calculating p-value: {e}")
    
    def AA_plot(self, data):
        AA_test_data = data[data['experiment'] == 'AA_test']
        control_group = AA_test_data[AA_test_data['group'] == 0].groupby('date')[self.conversion_metric].mean()
        treatment_group = AA_test_data[AA_test_data['group'] == 1].groupby('date')[self.conversion_metric].mean()
        exp_days = range(1, AA_test_data['date'].nunique() + 1)
        f, ax = plt.subplots(figsize=(10, 6))
        ax.plot(exp_days, control_group, label='Control', color='b')
        ax.plot(exp_days, treatment_group, label='Treatment', color='g')
        ax.set_xticks(exp_days)
        ax.set_title('AA Test')
        ax.set_ylabel('Convert Rate per Day')
        ax.set_xlabel('Days in the Experiment')
        ax.legend()
        return plt

    def validate_AA_test(self, data):
        p_value = self.AA_test(data)[0]
        try:
            if p_value < self.AA_alpha:
                return 'Significant difference found. Check for errors.'
            else:
                return 'No significant difference found. Proceed with AB test.'
        except Exception as e:
            print(f"Error validating AA test: {e}")

# Example usage

analyzer=ABTestAnalyzer(config_path)
analyzer.load_data()
# Call the functions and store the results
sample_size = analyzer.calculate_sample_size(analyzer.pretest_data)
test_duration = analyzer.test_duration(analyzer.pretest_data)[0]
start_date = analyzer.test_duration(analyzer.pretest_data)[1].strftime('%Y-%m-%d')
end_date = analyzer.test_duration(analyzer.pretest_data)[2].strftime('%Y-%m-%d')
sample_duration = analyzer.test_duration(analyzer.pretest_data)[3]
avg_conversion = analyzer.pretest_data[analyzer.conversion_metric].mean().round(4)
budget = analyzer.budget(analyzer.pretest_data)

# Convert the results to strings
sample_size_str = str(round(sample_size*(1+analyzer.group_ratio)))
test_duration_str = str(test_duration)
start_date_str = str(start_date)
end_date_str = str(end_date)
sample_duration_str = str(sample_duration)
budget_str = str(round(budget,2))

# AA test
# Check if the data contains 'AA_test'
if 'AA_test' in analyzer.pretest_data['experiment'].unique():
    AA_start = analyzer.AA_duration(analyzer.pretest_data)[0].strftime('%Y-%m-%d')
    AA_end = analyzer.AA_duration(analyzer.pretest_data)[1].strftime('%Y-%m-%d')
    AA_duration = analyzer.AA_duration(analyzer.pretest_data)[2]
    AA_control_mean = analyzer.AA_test(analyzer.pretest_data)[1].round(4)
    AA_treatment_mean = analyzer.AA_test(analyzer.pretest_data)[2].round(4)
    AA_pvalue= analyzer.AA_test(analyzer.pretest_data)[0]
    AA_plot= analyzer.AA_plot(analyzer.pretest_data)
    validate_AA_test = analyzer.validate_AA_test(analyzer.pretest_data)

    # Convert the results to strings
    AA_start_str = str(AA_start)
    AA_end_str = str(AA_end)
    AA_duration_str = str(AA_duration)
    AA_control_mean_str = str(AA_control_mean)
    AA_treatment_mean_str = str(AA_treatment_mean)
    p_value_str = str(round(AA_pvalue,3))
    validate_AA_test_str = str(validate_AA_test)
else:
    print("No AA_test data found. Skipping AA test.")



# Define the directory
directory = 'output'
# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the results to a file

with open(os.path.join(directory, 'AB_pretest.txt'), 'w') as f:
    f.write('Sample start date: ' + start_date_str + ', Sample end date: ' + end_date_str + ', duration: '+ sample_duration_str + '\n')
    f.write('Average conversion: ' + str(avg_conversion) + '\n')
    f.write('Sample size needed in total: ' + sample_size_str + '\n')
    f.write('Test duration needed: ' + test_duration_str + '\n')
    f.write('Budget needed in total: ' + budget_str + '\n')
    
    if 'AA_test' in analyzer.pretest_data['experiment'].unique():
        f.write('AA test start date: ' + AA_start_str + ', AA test end date: ' + AA_end_str + ', duration: '+ AA_duration_str + '\n')
        f.write('AA control average conversion: ' + AA_control_mean_str + ', AA treatment average conversion: ' + AA_treatment_mean_str + '\n')
        f.write('AA significance level: ' + str(analyzer.AA_alpha) + ', AA test p-value: ' + p_value_str +'\n')
        f.write('AA test validation: ' + validate_AA_test_str + '\n')
        # Save the plot in the directory
        plt.savefig(os.path.join(directory, 'AA_test.png'))

# Create a new Document
doc = Document()
# Add the string to the document
doc.add_heading('AB Test Pretest Results',0)
doc.add_heading('Sample Data',1)
doc.add_paragraph('Sample start date: ' + start_date_str + ', Sample end date: ' + end_date_str + ', duration: '+ sample_duration_str)
doc.add_paragraph('Average conversion: ' + str(avg_conversion))
doc.add_heading('Power Analysis, Test Duration and Budget',1)
doc.add_paragraph('Sample size needed in total: ' + sample_size_str)
doc.add_paragraph('The test duration needed: ' + test_duration_str)
doc.add_paragraph('Budget needed in total: ' + budget_str)
if 'AA_test' in analyzer.pretest_data['experiment'].unique():
    doc.add_heading('Pretest Validation: AA Test',1)
    doc.add_paragraph('AA test start date: ' + AA_start_str + ', AA test end date: ' + AA_end_str + ', duration: '+ AA_duration_str)
    doc.add_paragraph('AA control average conversion: ' + AA_control_mean_str + ', AA treatment average conversion: ' + AA_treatment_mean_str)
    doc.add_paragraph('AA significance level: ' + str(analyzer.AA_alpha) + ', AA test p-value: ' + p_value_str)
    doc.add_paragraph('AA test validation: ' + validate_AA_test_str)
    # Save the plot in the directory
    plot_path = os.path.join(directory, 'AA_test.png')
    # Add the plot to the document
    doc.add_picture(plot_path, width=Inches(5.0))
    # Save the document in the directory
    doc.save(os.path.join(directory, 'AB_pretest.docx'))
else:
    doc.save(os.path.join(directory, 'AB_pretest.docx'))
