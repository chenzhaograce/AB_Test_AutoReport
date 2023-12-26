# AB Test AutoReport 🤖

## Description 🧩
`ABTestAnalyzer` is a Python class for performing A/B testing analysis. The class includes methods for sample size calculation, test duration calculation, calculating effect size, performing AA test, AB test, Normality check, Homogenity check, Novelty effect test, Chi-square tests, and more.  
The class uses a configuration file in YAML format to set various parameters for the analysis.

## Navigating the Project 🗺️
To navigate to the project directory after cloning the repository, you can use the `cd` command in your terminal. Here's how you can do it:
1. Open your terminal.
2. Use the `cd` command followed by the path of your directory to navigate to the project. For example:
```bash
cd path/to/your/directory/AB_Test_AutoReport
```
3. If you want to know your current directory location, you can use the pwd (Print Working Directory) command:
```pwd```

## Installation 🔌
- clone the repo `https://github.com/chenzhaograce/AB_Test_AutoReport.git`
- Make sure Python and pip are installed
- Install dependencies with `pip3 install -r requirements.txt`
- Change parameters in configuration file `config.yaml`
- Run `python3 AB_pretest.py` to run the initial test of AB testing
- Run `python3 AB_post-test.py` to run the after test of AB testing
- Check the auto reports of .txt and .png files generated under `/output` folder
- Check the logs under `/logs` folder

## AB test one-stop automation 🎯
The `ABTestAnalyzer` class provides a comprehensive suite of methods for A/B testing analysis:
### Pretest
- **Missing Value Check**: The class includes a method for checking missing values in the dataset. It will provide a report of the number and percentage of missing values in each column.
- **Outliers Check**: The class includes a method for detecting outliers in the dataset. It uses the IQR method to detect outliers and provides a report of the number and percentage of outliers in each column.
- **Data Loading and Effect Size Calculation**: The class includes methods for loading data and calculating effect size.
- **Sample Size Calculation**: The class includes a method for calculating the required sample size based on the desired power and significance level.
- **Test Duration Calculation**: The class includes a method for calculating the required test duration based on the desired sample size and the expected traffic. 
- **Budget Need for Test**: The class includes a method for estimating the budget needed for the test based on the sample size, expected traffic, and average spend per observation.
- **Chi-square Test**: The class includes a method for performing a Chi-square test to compare the conversion rates of the two groups.
- **AA Test**: The class includes a method for performing an AA test to check if the two groups are identical before the actual A/B test.
### Post test
- **Normality Test**: The class includes a method for testing if the data follows a normal distribution.
- **Homogeneity Test**: The class includes a method for testing if the variances of two or more groups are equal.
- **SRM Test**: The class includes a method for performing a Sequential Ratio Method (SRM) test to detect any significant changes over time.
- **Novelty Effect Analysis**: The class includes a method for analyzing the novelty effect, i.e., the initial surge in performance due to the novelty of the new version.
- **AB Test**: The class includes a method for performing the actual A/B test, comparing the performance of the two versions.
- **Confidence Interval Calculation**: The class includes a method for calculating the confidence interval of the difference in performance between the two versions.
- **Absolute Lift Calculation**: The class includes a method for calculating the absolute lift, i.e., the absolute difference in performance between the two versions.
- **Relative Lift Calculation**: The class includes a method for calculating the relative lift, i.e., the percentage increase in performance of the new version compared to the old version.
### Generate conclusion
- Based on the performance, decide wheter or not to use the new version of AB test.


## Features 🏵️
- **Automated Analysis**: The `ABTestAnalyzer` class automates the process of performing A/B testing analysis. Just run the scripts and get your results.
- **Configurable**: The class uses a configuration file in YAML format, allowing you to easily adjust various parameters for the analysis.
- **Comprehensive**: The class includes methods for sample size calculation, test duration calculation, calculating effect size, performing AA test, AB test, Novelty effect test, Chi-square tests, and more.
- **Visualizations**: The class generates visualizations of the test results, which are saved as .png files in the `/output` folder.
- **Report Generation**: The class generates a report of the test results, which is saved as a .txt file in the `/output` folder.
- **Pretest and Posttest Analysis**: The repository includes scripts for both pretest (`AB_pretest.py`) and posttest (`AB_post-test.py`) analysis.
- **Easy Installation**: The repository includes a `requirements.txt` file for easy installation of dependencies.

## Data Preparation 📊
Before using the `ABTestAnalyzer` class, you need to prepare your data in the following format:
- The data should be on an ID level, meaning each row corresponds to a unique ID. The ID data should be unique and not duplicated. The data should be in two separate CSV files (If pretest and post-test are in one file, you can also use the same.): one for the pretest data and one for the posttest data.
- Each file should include the following columns:
    - **Date**: The date of the observation.
    - **ID**: The ID of the observation (this could be a user_id, impression_id, etc.).
    - **Conversion**: The conversion metric. This can be either binary (1 for a conversion, 0 for no conversion) or continuous (sales, signups, etc.). The type of conversion metric should be specified in the configuration file.
    - **Experiment**: The name of the test ('ab_test','AA_test','landingpage_test', etc.).
    - **Group**: The group of the observation (0 for "control", 1 for "treatment").
    - **Spend** (Optional): The spend for the observation, if budget calculation is needed.
    - etc.
- The names of these columns should be specified in the configuration file.

Here's an example of how your data might look:

| Date       | ID  | Conversion | Experiment | Group | Spend |
|------------|-----|------------|------------|-------|-------|
| 2021-01-01 | 1   | 1          | Test1      | 0     | 100   |
| 2021-01-02 | 2   | 0          | Test1      | 1     | 150   |
| 2021-01-03 | 3   | 1          | Test1      | 0     | 200   |
| 2021-01-04 | 4   | 0          | Test1      | 1     | 250   |
| ...        | ... | ...        | ...        | ...   | ...   |

## Contributing 🤝
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/chenzhaograce/AB_Test_AutoReport/issues). You can also take a look at the [contributing guide](https://github.com/chenzhaograce/AB_Test_AutoReport/blob/main/CONTRIBUTING.md).

## License 📝
This project is [MIT](https://github.com/chenzhaograce/AB_Test_AutoReport/blob/main/LICENSE) licensed.