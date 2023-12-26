
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm
import statsmodels.stats.api as sms
import math
import os
from docx import Document
from docx.shared import Inches
import logging
import yaml
from datetime import datetime, timedelta


class ABTestAnalyzer:
    def __init__(self):
        # Initialize the log attribute
        self.log = logging.getLogger(__name__)
        self.setupLogger()        
        try:
            # Specify the config_path directly
            config_path = 'config.yaml'            
            # Get the absolute path of the config file
            abs_config_path = os.path.abspath(config_path)
            with open(abs_config_path, 'r') as file:
                config = yaml.safe_load(file)

            self.pretest_path = config['pretest_path']
            self.conversion_metric = config['conversion_metric']
            self.id_column = config['id_column']
            self.date_column = config['date_column']
            self.spend_column = config.get('spend_column', None)  # Default to None if 'spend_column' is not found
            self.group_column = config['group_column']
            self.control_group = config['control_group']
            self.treatment_group = config['treatment_group']
            self.experiment_column = config['experiment_column']
            self.binary = config['binary']
            self.MDE = config['MDE']
            self.significance_level = config['significance_level']
            self.power = config['power']
            self.group_ratio = config['group_ratio']
            self.AA_alpha = config['AA_alpha']
            self.AA_metric = config['AA_metric']
            self.pretest_data = None
            self.log.info("ABTestAnalyzer initialized successfully.")
            self.log.info(f"Pretest path: {self.pretest_path}, Conversion metric: {self.conversion_metric}, ID column: {self.id_column}, Date column: {self.date_column}, Experiment column: {self.experiment_column}, Spend column: {self.spend_column}, Proportion: {self.binary}, MDE: {self.MDE}, Significance level: {self.significance_level}, Power: {self.power}, Group ratio: {self.group_ratio}, AA alpha: {self.AA_alpha}, AA metric: {self.AA_metric}")      
        except Exception as e:
            self.log.error(f"Error initializing ABTestAnalyzer: {e}")
            raise e
    
    def setupLogger(self):
        dt = datetime.strftime(datetime.now(), "%m_%d_%y %H_%M_%S ")

        if not os.path.isdir('./logs'):
            os.mkdir('./logs')

        logging.basicConfig(filename=('./logs/' + str(dt) + 'AB_pretest.log'), filemode='w',
                            format='%(asctime)s::%(name)s::%(levelname)s::%(message)s', datefmt='./logs/%d-%b-%y %H:%M:%S')
        self.log.setLevel(logging.DEBUG)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
        c_handler.setFormatter(c_format)
        self.log.addHandler(c_handler)    

    def load_data(self):
        try:
            self.pretest_data = pd.read_csv(self.pretest_path, low_memory=False)
            self.pretest_data[self.date_column] = pd.to_datetime(self.pretest_data[self.date_column], format='%m/%d/%y')
        except Exception as e:
            self.log.error(f"Error loading data: {e}")
            raise e
    
    def check_missing(self, data):
        try:
            # missing data and missing percentage in id_column, date_column and conversion_metric
            df=data[[self.id_column, self.date_column, self.conversion_metric]]
            missing_data = df.isnull().sum()
            missing_data_percent = (df.isnull().sum() / len(df)) * 100
            missing_data_df = pd.DataFrame({'Total Missing': missing_data, 'Percentage': missing_data_percent})
            return missing_data_df[missing_data_df['Total Missing'] > 0]
        except Exception as e:
            self.log.error(f"Error checking missing values: {e}")
            raise e
    
    def check_outliers(self, data):
        try:
            # check if there are outliers in the conversion_metric by date
            conversions=data.groupby(self.date_column)[self.conversion_metric].sum().reset_index().rename(columns={0: self.conversion_metric, 'index': self.date_column})
            q1 = conversions[self.conversion_metric].quantile(0.25)
            q3 = conversions[self.conversion_metric].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = conversions[(conversions[self.conversion_metric] < lower_bound) | (conversions[self.conversion_metric] > upper_bound)]
            return outliers    
        except Exception as e:
            self.log.error(f"Error checking outliers: {e}")
            raise e

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
            self.log.error(f"Error calculating effect size: {e}")
            raise e

    def calculate_sample_size(self, data):
        binary = self.binary
        try: 
            effect_size = self.calculate_effect_size(data)
            sample_size = tt_ind_solve_power(effect_size=effect_size, alpha=self.significance_level, power=self.power, ratio=self.group_ratio)
            return sample_size
        except Exception as e:
            self.log.error(f"Error calculating sample size: {e}")
            raise e

    def test_duration(self, data):
        try: 
            sample_size = self.calculate_sample_size(data)
            daily_unique_id = data.groupby(self.date_column)[self.id_column].nunique()
            start_date = daily_unique_id.index.min()
            end_date = daily_unique_id.index.max()
            sample_duration = (end_date - start_date).days + 1
            avg_daily_unique = daily_unique_id.mean()
            test_duration = sample_size * (1 + self.group_ratio) / avg_daily_unique
            adjusted_test_duration = math.ceil(test_duration / 7) * 7
            return adjusted_test_duration, start_date, end_date, sample_duration
        except Exception as e:
            self.log.error(f"Error calculating test duration: {e}")
            raise e
         
    def budget(self, data):
        try: 
            sample_size = self.calculate_sample_size(data)
            spend = data[self.spend_column].sum()
            count = data[self.id_column].nunique()
            avg_cpm = spend / count
            budget = avg_cpm * sample_size * (1 + self.group_ratio)
            return budget
        except Exception as e:
            self.log.error(f"Error calculating budget: {e}")
            raise e

    def AA_duration(self, data):
        AA_test_data = data[data[self.experiment_column] == self.AA_metric]
        try:
            start_date = AA_test_data[self.date_column].min()
            end_date = AA_test_data[self.date_column].max()
            duration = (end_date - start_date).days + 1
            return start_date, end_date, duration
        except Exception as e:
            self.log.error(f"Error calculating AA test duration: {e}")
            raise e          

    def AA_test(self, data):
        AA_test_data = data[data[analyzer.experiment_column] == self.AA_metric]
        control_group = AA_test_data[AA_test_data[self.group_column] == self.control_group][self.conversion_metric]
        treatment_group = AA_test_data[AA_test_data[self.group_column] == self.treatment_group][self.conversion_metric]
        try: 
            if data[self.conversion_metric].nunique() == 2:
                _, p_value, _ = proportions_chisquare([control_group.sum(), treatment_group.sum()], nobs=[control_group.count(), treatment_group.count()])
            else:
                _, p_value, _ = ttest_ind(control_group, treatment_group, usevar='unequal')

            return p_value, control_group.mean(), treatment_group.mean()
        except Exception as e:
            self.log.error(f"Error calculating AA test: {e}")
            raise e
    
    def AA_plot(self, data):
        try:
            AA_test_data = data[data[analyzer.experiment_column] == self.AA_metric]
            control_group = AA_test_data[AA_test_data[self.group_column] == self.control_group].groupby(self.date_column)[self.conversion_metric].mean()
            treatment_group = AA_test_data[AA_test_data[self.group_column] == self.treatment_group].groupby(self.date_column)[self.conversion_metric].mean()
            exp_days = range(1, AA_test_data[self.date_column].nunique() + 1)
            f, ax = plt.subplots(figsize=(12, 6))
            # plt.style.use('fivethirtyeight') # optional
            ax.plot(exp_days, control_group, label='Control', color='b', marker='o')
            ax.plot(exp_days, treatment_group, label='Treatment', color='r' ,marker='o')
            ax.set_xticks(exp_days)
            ax.set_title('AA Test')
            ax.set_ylabel('Convert Rate per Day')
            ax.set_xlabel('Days in the Experiment')
            ax.legend()
            return plt
        except Exception as e:
            self.log.error(f"Error plotting AA test: {e}")
            raise e

    def validate_AA_test(self, data):
        p_value = self.AA_test(data)[0]
        try:
            if p_value < self.AA_alpha:
                return 'Significant difference found. Check for errors.'
            else:
                return 'No significant difference found. Proceed with AB test.'
        except Exception as e:
            self.log.error(f"Error validating AA test: {e}")
            raise e

# Example usage
analyzer=ABTestAnalyzer()
analyzer.load_data()
# Call the functions and store the results
sample_size = analyzer.calculate_sample_size(analyzer.pretest_data)
test_duration = analyzer.test_duration(analyzer.pretest_data)[0]
start_date = analyzer.test_duration(analyzer.pretest_data)[1].strftime('%Y-%m-%d')
end_date = analyzer.test_duration(analyzer.pretest_data)[2].strftime('%Y-%m-%d')
sample_duration = analyzer.test_duration(analyzer.pretest_data)[3]
avg_conversion = analyzer.pretest_data[analyzer.conversion_metric].mean()
data_columns=analyzer.pretest_data.columns.values
data_shape=analyzer.pretest_data.shape
if analyzer.spend_column is not None:
    budget = analyzer.budget(analyzer.pretest_data)
else:
    budget = None  # or some default value

# AA test
# Check if the data contains 'AA_test'
if 'AA_test' in analyzer.pretest_data[analyzer.experiment_column].unique():
    AA_start = analyzer.AA_duration(analyzer.pretest_data)[0].strftime('%Y-%m-%d')
    AA_end = analyzer.AA_duration(analyzer.pretest_data)[1].strftime('%Y-%m-%d')
    AA_duration = analyzer.AA_duration(analyzer.pretest_data)[2]
    AA_control_mean = analyzer.AA_test(analyzer.pretest_data)[1]
    AA_treatment_mean = analyzer.AA_test(analyzer.pretest_data)[2]
    AA_pvalue= analyzer.AA_test(analyzer.pretest_data)[0]
    AA_plot= analyzer.AA_plot(analyzer.pretest_data)
    validate_AA_test = analyzer.validate_AA_test(analyzer.pretest_data)

else:
    print("No AA_test data found. Skipping AA test.")

# Define the directory
directory = 'output'
# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the results to a file
with open(os.path.join(directory, 'AB_pretest_report.txt'), 'w') as f:
    analyzer.log.info(f"Writing results to {os.path.join(directory, 'AB_pretest_report.txt')}")
    f.write('# AB Test Pretest Report\n')
    f.write(f'\n## Data Quality Check:\n')
    if analyzer.check_missing(analyzer.pretest_data).empty:
        f.write('No missing values found.\n')
    else:
        f.write(f'Missing values: {analyzer.check_missing(analyzer.pretest_data)}\n')
    if analyzer.check_outliers(analyzer.pretest_data).empty:
        f.write('No outliers found.\n')
    else:
        f.write(f'Outliers: {analyzer.check_outliers(analyzer.pretest_data)}\n')
    f.write(f'\n## Pretest Parameters:\n')
    f.write(f'Minimum Detectable Effect: {analyzer.MDE}\n')
    f.write(f'Significance Level: {analyzer.significance_level}\n')
    f.write(f'Power: {analyzer.power}\n')
    f.write(f'\n## Sample Data Preparation:\n')
    f.write(f'Number of rows: {data_shape[0]}, Number of columns: {data_shape[1]}\n')
    f.write(f'Data columns: {data_columns}\n')
    f.write(f'Conversion metric: {analyzer.conversion_metric}\n')
    f.write(f'Sample start date: {start_date}, Sample end date: {end_date} , duration: {sample_duration} days\n')
    f.write(f'Average conversion: {avg_conversion:.4f}\n')
    f.write('\n## Power Analysis, Test Duration and Budget:\n')
    f.write(f'Sample size needed in total: {sample_size*2:.0f}\n')
    f.write(f'Test duration needed: {test_duration} days\n')
    if analyzer.spend_column is not None:
        f.write(f'Budget needed in total: {budget:.2f}\n')
    else:
        pass
    
    if 'AA_test' in analyzer.pretest_data[analyzer.experiment_column].unique():
        f.write('\n## Pretest Validation - AA Test:\n')
        f.write(f'AA test start date: {AA_start}, AA test end date: {AA_end}, duration: {AA_duration} days\n')
        f.write(f'AA control average conversion: {AA_control_mean:.4f}, AA treatment average conversion: {AA_treatment_mean:4f}\n')
        f.write(f'AA significance level: {analyzer.AA_alpha}, AA test p-value {AA_pvalue:.3f}\n')
        f.write(f'AA test validation: {validate_AA_test}\n')
        # Save the plot in the directory
        plt.savefig(os.path.join(directory, 'AA_test.png'))

