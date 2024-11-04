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
   git clone <repository_url>
   cd <repository_name>
   ```
2. **Install requirements**:

   ```  
   pip install -r requirements.txt
   ```
 
