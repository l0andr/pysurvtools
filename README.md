# pysurvtools
A set of Python tools for survival analysis and data preparation for survival analysis.

## Contents

* <b>survplots.py</b> - visualization and analysis tool built using the lifelines library for survival analysis (Davidson-Pilon, 2019). It provides easy-to-generate plots and analyses for exploring survival data, including Kaplan-Meier survival curves, value counts, histograms, and Fisher's exact tests to assess relationships between binary factors and outcomes. The tool supports custom legends, data pre-filtering, and group size adjustments, making it highly adaptable for large survival studies. Outputs include various plot types and summaries that are automatically compiled into a PDF report, offering a streamlined workflow for robust survival analysis and reporting. <br>
* <b>cox-analysis.py</b> - tool, built around the lifelines library for survival analysis (Davidson-Pilon, 2019), performs survival prediction using the Cox Proportional Hazards model with support for penalization parameter tuning. The tool allows for a comprehensive analysis pipeline, including grid search optimization of penalization parameters (L1/L2 ratio) and univariate analysis to identify significant predictors. Model quality is evaluated through metrics like concordance index, log-likelihood, log-rank test, AIC, and survival probability calibration. This tool outputs detailed visual reports and summaries of significant factors influencing survival, as well as model performance plots, and can generate tailored PDF reports for streamlined survival analysis interpretation. <br>
* <b>oncoplot.py</b> -  tool is a wrapper around the pyoncoprint library (https://pubmed.ncbi.nlm.nih.gov/37037472/), designed to create customizable oncoprints for mutation data visualization. It enables users to specify genes and clinical factors of interest, with options for gene sorting and detailed annotations, including stage formatting in Roman numerals and response classifications. This tool provides flexibility for mutation markers and color-coded clinical annotations, and outputs can be saved as PDF or PNG.  <br>

## Instalation

### Pre-requirements

Before installing, ensure you have the following:

- **Python 3.9 or later**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Bash-like shell**: A shell environment for running commands. This could be Git Bash (Windows), Terminal (macOS), or any Linux shell.

### Installation

1. **Clone the repository**:

   Open your shell and run:
   ```bash
   git clone https://github.com/l0andr/pysurvtools.git
   cd pysurvtools
   ```
2. **Install requirements**:

   ```  
   pip install -r requirements.txt
   ```
## Tools description

### Survplots

<table>
   <tr>
      <td>
         <img src= "img/survplots_exmp1.png" >
      </td>
      <td>
         <img src= "img/survplots_exmp2.png" >
      </td>
   </tr>
</table>

Usage example:
```
python survplots.py --input_csv tdf.csv --survival_time_col disease_free_time --plot kaplan_meier --max_survival_length 2000 --columns tnum,response,sex,cancer_type,alcohol_history,drugs,anatomic_stage,gene_FGF4,gene_CDKN2A,gene_MYL1,gene_ARID2 --output_pdf $output_dir/figure_6_disease_free_time_kaplan_meier.pdf --min_size_of_group 0.01 --custom_legend km_legend.json --filter_nan_columns treatment_type,response

```


<details>
     <summary>Parameters and options of survplots</summary>

| Option                   | Description                                                                                                                                                                         | Type     | Default Value       |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------------|
| `--input_csv`            | Path to the input CSV file containing survival data. This file should include relevant columns, such as patient IDs, event status, and survival time.                               | `str`    | **Required**        |
| `--output_pdf`           | Path to the output PDF file where all generated figures will be saved.                                                                                                             | `str`    | **Required**        |
| `--plot`                 | Type of plot to generate. Options include: `kaplan_meier`, `pieplots`, `floathistograms`, `valuecounts`, and `fisher_exact_test`.                                                  | `str`    | `"kaplan_meier"`    |
| `--status_col`           | Column name for the event status indicator (e.g., whether an event, such as death or relapse, has occurred).                                                                       | `str`    | `"status"`          |
| `--survival_time_col`    | Column name for survival time, usually recorded in days.                                                                                                                           | `str`    | `"survival_in_days"`|
| `--patient_id_col`       | Column name for patient IDs, useful for linking observations.                                                                                                                      | `str`    | `"patient_id"`      |
| `--columns`              | One or more specific columns to include in the plot. Separate multiple columns with commas. Supports wildcard `*` at the end to include columns starting with a specific prefix. | `str`    | `""`                |
| `--min_size_of_group`    | Minimum group size for Kaplan-Meier plots, defined as a fraction of all cases. Helps exclude small groups from analysis.                                                            | `float`  | `0.07`              |
| `--max_amount_of_groups` | Maximum number of groups per factor to display. Ensures the plot remains readable by limiting the number of groups.                                                                 | `int`    | `10`                |
| `--max_survival_length`  | Maximum time interval (in days) to consider for Kaplan-Meier plots. Any survival times beyond this will be truncated to the specified length.                                      | `float`  | `1825` (5 years)    |
| `--show`                 | If set, displays plots interactively in addition to saving them to the PDF.                                                                                                        | `flag`   | `False`             |
| `--verbose`              | Verbose level for logging output, where `0` is silent and higher numbers increase the level of detail.                                                                             | `int`    | `1`                 |
| `--custom_legend`        | Path to a JSON file containing custom legends for Kaplan-Meier plot labels. The JSON format should define group labels for each column used in the plot.                           | `str`    | `None`              |
| `--filter_nan_columns`   | Comma-separated list of columns in which NaN values will be filtered out. This helps ensure that missing data in these columns does not interfere with plotting.                   | `str`    | `""`                |
| `--title`                | Title for the plot, which will appear in the output figures.                                                                                                                       | `str`    | `""`                |

#### Plot Types

SurvPlots supports a variety of plot types, each tailored for different aspects of survival data visualization:

1. **Kaplan-Meier Plot (`kaplan_meier`)**  
   Generates Kaplan-Meier survival curves for specified groups, useful for comparing survival distributions across categories or treatment groups. This plot type supports grouping by categorical variables and allows custom legends to clarify group labels.

2. **Pie Charts (`pieplots`)**  
   Creates pie charts for categorical variables, providing an intuitive visualization of the distribution across categories. This plot is helpful for understanding the proportion of different categories within the dataset.

3. **Histograms (`floathistograms`)**  
   Plots histograms for continuous (float) variables, displaying their distribution across bins. Median values are annotated in each plot to provide a summary of the central tendency.

4. **Value Counts (`valuecounts`)**  
   Produces bar plots showing the count of each unique value in the specified columns, with percentages labeled on the bars. This plot is ideal for categorical variables, offering a clear representation of the frequency distribution.

5. **Fisher’s Exact Test (`fisher_exact_test`)**  
   Conducts Fisher’s exact test on specified binary factors and outputs a scatter plot with odds ratios and p-values. Significant associations are highlighted, with p-values and odds ratios clearly labeled, aiding in the identification of potentially important relationships between factors.

---
</details>

### Cox_analysis

<table>
   <tr>
      <td>
         <img src= "img/cox_analysis_exmp1.png" >
      </td>
      <td>
         <img src= "img/cox_analysis_exmp2.png" >
      </td>
   </tr>
</table>

Usage example:
```
python cox_analysis.py -input_csv 2024_transformed.csv --genes "" --factors sex,age,p16,smoking,race,cancer_type,prior_cancer,drugs,treatment_type0,total_mutations,anatomic_stage,msi_status,tmb_level,lvi,pni,smoking_packs,pdl1_category,response_0,alcohol_history --penalizer 0.01 --l1ratio 0.01 --univar "" $show --model_report $output_dir/figure_3_overall_survival_cox_univariant_factors.pdf --title "overall survival time "
```


<details>
     <summary>Parameters and options of cox_analysis</summary>
     ## Cox-Analysis Tool Options

| Option                   | Description                                                                                                                                                                       | Type     | Default Value         |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-----------------------|
| `-input_csv`             | Path to the input CSV file containing survival data. This file should include columns like patient IDs, event status, survival time, and relevant covariates.                     | `str`    | **Required**          |
| `--min_cases`            | Minimum number of cases required for mutation columns. Genes with fewer cases are excluded from the analysis.                                                                     | `int`    | `3`                   |
| `--genes`                | Comma-separated list of genes of interest for the analysis.                                                                                                                       | `str`    | `None`                |
| `--factors`              | Comma-separated list of additional factors of interest to be included in the model.                                                                                               | `str`    | `None`                |
| `--status_col`           | Column name representing the event status (e.g., death or relapse).                                                                                                              | `str`    | `"status"`            |
| `--survival_time_col`    | Column name representing survival time, typically recorded in days.                                                                                                              | `str`    | `"survival_in_days"`  |
| `--patient_id_col`       | Column name for patient IDs to ensure observations are properly tracked.                                                                                                          | `str`    | `"patient_id"`        |
| `--calib_t0`             | Time used for model calibration in survival probability calculations.                                                                                                             | `int`    | `1900`                |
| `--l1ratio`              | Ratio of L1 regularization in the Cox model’s penalty. If negative, a grid search is performed to optimize this value.                                                            | `float`  | `-1`                  |
| `--penalizer`            | Penalization parameter for the Cox model optimizer. If negative, a grid search is performed to find the optimal value.                                                            | `float`  | `-1`                  |
| `--opt_report`           | Path for saving the optimization report PDF, detailing results from grid search and parameter tuning.                                                                             | `str`    | `"cox_optim_report.pdf"` |
| `--model_report`         | Path for saving the model report PDF, including model summary and statistical tests.                                                                                              | `str`    | `"cox_model_report.pdf"`  |
| `--univar`               | If set, performs univariate analysis on specified factors, varying each factor in isolation.                                                                                      | `str`    | `None`                |
| `--verbose`              | Verbose level for logging output, where `0` is silent and higher numbers increase the level of detail.                                                                            | `int`    | `1`                   |
| `--show`                 | If set, displays plots interactively in addition to saving them to the PDF.                                                                                                       | `flag`   | `False`               |
| `--plot_outcome`         | If set, generates and displays plots of partial effects on the outcome.                                                                                                          | `flag`   | `False`               |
| `--filter_nan`           | If set, removes rows with NaN values across gene and factor columns.                                                                                                              | `flag`   | `False`               |
| `--filter_nan_columns`   | Comma-separated list of columns to filter out NaN values from, ensuring completeness in specific variables.                                                                      | `str`    | `""`                  |
| `--title`                | Title for the plot, which will appear in the output figures.                                                                                                                      | `str`    | `""`                  |

#### Cox-Analysis Plot Types and Outputs

The **Cox-Analysis** tool, built around the **lifelines** library for survival analysis ([Davidson-Pilon, 2019](https://doi.org/10.21105/joss.01317)), supports various plots and statistical summaries for analyzing survival data:

1. **Univariate Analysis Plot**  
   Conducts univariate Cox regression for each specified factor individually, allowing visualization of the hazard ratios and confidence intervals. This plot highlights factors with significant effects on survival, helping to identify potential predictors for further analysis.

2. **Grid Search Optimization Heatmaps**  
   When L1 ratio and penalizer values are set to `-1`, the tool performs a grid search and generates heatmaps of model performance metrics (e.g., concordance index, log-likelihood) across different parameter combinations. These plots provide insights into optimal penalization settings for the Cox model.

3. **Model Summary and Tree Plot**  
   Once the model is fit, the tool generates a forest plot of significant factors with hazard ratios and confidence intervals. This summary plot visually conveys the risk associated with each factor, aiding in the interpretation of multivariate results.
 
</details>        
