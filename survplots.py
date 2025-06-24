import argparse
import json
from copy import deepcopy
from os.path import split
from tabnanny import verbose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import fisher_exact

import scipy.stats as stats

def plot_kruskal_wallis_boxplot(df:pd.DataFrame, split_column:str, value_column:str, legend_dict ={},title ="",xlabel = "",ylabel = "", fontsize=14):
    fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1)
    values_lists =[]
    split_list = sorted(df[split_column].unique().tolist())
    for v in split_list:
        values_lists.append(df[df[split_column] == v][value_column].values)
    
    # Create boxplot
    bp = plt.boxplot(values_lists)
    
    # Set x-axis labels
    ticks_str = []
    if split_column in legend_dict:
        for x in split_list:
            if str(x) in legend_dict[split_column]:
                ticks_str.append(legend_dict[split_column][str(x)])
            else:
                ticks_str.append(str(x))
    else:
        ticks_str = [str(x) for x in split_list]
    plt.xticks(np.arange(1,len(split_list)+1), ticks_str, fontsize=fontsize)

    # Perform Kruskal-Wallis H test
    kwht_dft = stats.kruskal(*values_lists)
    
    # Perform pairwise Mann-Whitney U tests
    pairwise_pvalues = {}
    n_groups = len(split_list)
    
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            group1_name = str(split_list[i])
            group2_name = str(split_list[j])
            pair_key = f"{group1_name}_vs_{group2_name}"
            
            # Perform Mann-Whitney U test
            try:
                stat, p_value = stats.mannwhitneyu(values_lists[i], values_lists[j], 
                                                 alternative='two-sided')
                pairwise_pvalues[pair_key] = p_value
            except ValueError:
                # Handle cases where one group has all identical values
                pairwise_pvalues[pair_key] = 1.0

    # Add significance indicators on the plot
    y_max = max([max(vals) for vals in values_lists if len(vals) > 0])
    y_min = min([min(vals) for vals in values_lists if len(vals) > 0])
    y_range = y_max - y_min
    
    # Define significance levels
    sig_levels = [0.001, 0.01, 0.05]
    sig_symbols = ['***', '**', '*']
    
    # Plot significance bars
    bar_height = y_range * 0.05
    current_y = y_max + y_range * 0.1
    
    for pair_key, p_value in pairwise_pvalues.items():
        group1, group2 = pair_key.split('_vs_')
        idx1 = split_list.index(float(group1) if group1.replace('.', '').replace('-', '').isdigit() else group1)
        idx2 = split_list.index(float(group2) if group2.replace('.', '').replace('-', '').isdigit() else group2)
        
        # Determine significance symbol
        sig_symbol = ''
        for level, symbol in zip(sig_levels, sig_symbols):
            if p_value < level:
                sig_symbol = symbol
                break
        
        if sig_symbol:
            # Draw significance bar
            x1, x2 = idx1 + 1, idx2 + 1
            plt.plot([x1, x1, x2, x2], [current_y, current_y + bar_height, current_y + bar_height, current_y], 
                    'k-', linewidth=1)
            
            # Add significance symbol and p-value
            symbol_text = f"{sig_symbol}\np={p_value:.4E}"
            plt.text((x1 + x2) / 2, current_y + bar_height + y_range * 0.01, symbol_text, 
                    ha='center', va='bottom', fontsize=fontsize-2, fontweight='bold')
            current_y += y_range * 0.2

    # Set labels and title
    if ylabel != "":
        ylabel_str = ylabel
    else:
        ylabel_str = value_column
    plt.ylabel(ylabel_str, fontsize=fontsize)
    
    if xlabel != "":
        xlabel_str = xlabel
    else:
        if split_column in legend_dict and 'legend name' in legend_dict[split_column]:
            xlabel_str = legend_dict[split_column]['legend name']
        else:
            xlabel_str = split_column
    
    if title != "":
        title_str = title
    else:
        title_str = f"{ylabel_str} and {xlabel_str}"
    
    # Create title with only Kruskal-Wallis p-value
    title_with_pvalues = title_str + f"\nKruskal-Wallis H Test p-value: {kwht_dft[1]:.4E}"
    
    plt.title(title_with_pvalues, fontsize=fontsize)
    plt.xlabel(xlabel_str, fontsize=fontsize)
    
    # Set y-axis tick label fontsize
    plt.yticks(fontsize=fontsize)
    
    # Adjust y-axis limits to accommodate significance bars
    plt.ylim(y_min - y_range * 0.1, current_y + y_range * 0.1)
    plt.tight_layout()
    return fig, ax

def plot_piecharts_of_categorial_variables(df_clean:pd.DataFrame):
    #df_tmp = df_clean.loc[:, ~df_clean.columns.str.contains('date', case=False)]
    #df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('gene_', case=False)]
    df_tmp = df_clean.loc[:, df.dtypes == 'category' ]
    nrows = min([3, len(df_tmp.columns)])
    ncols = int(np.ceil(len(df_tmp.columns) / nrows))
    # plot in one figure pie charts of all columns in df_tmp
    fig, ax = plt.subplots(figsize=(18, 10), nrows=nrows, ncols=ncols)
    if nrows == 1:
        ax = [ax]
    for i, col in enumerate(df_tmp.columns):
        df_tmp[col].value_counts(dropna=False).plot.pie(ax=ax[i // ncols, i % ncols], autopct='%.2f', fontsize=10)
        ax[i // ncols, i % ncols].set_title(f'{col}')
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(ax[j // ncols, j % ncols])
    plt.tight_layout()
    return fig, ax

def plot_value_counts(df, columns):
    fig, axes = plt.subplots(nrows=len(columns), figsize=(10, 6))
    if len(columns) == 1:
        axes = [axes]
    i = 0
    for col in columns:
        # Get value counts
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        labels_nan = counts.index
        values = counts.values
        labels = []
        for l in labels_nan:
            if not isinstance(l,str):
                labels.append('no data')
            else:
                labels.append(l)
        # Plot
        ax = axes[i]
        bars = ax.bar(labels, values)
        ax.set_title(f"Value сounts for {col}")
        ax.set_ylabel("Number of Occurrences")
        ax.set_xticks(np.linspace(0,len(labels)-1,len(labels)))
        ax.set_xticklabels(labels, rotation=5, ha='right')

        # Add percentage on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height} ({height/total:.2%})",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        ha='center', va='bottom',
                        xytext=(0, 10),
                        textcoords='offset points')
        i = i+1
    plt.tight_layout()
    return fig, axes


def plot_histograms_of_float_values(df_clean:pd.DataFrame):
    # plot in one figure histogram of all float columns with number of unique values more than sqrt(len(df.index))
    df_tmp = df_clean.loc[:, df_clean.dtypes == np.float64]
    #df_tmp = df_tmp.loc[:, df_tmp.nunique() > np.sqrt(len(df_clean.index))/2]
    # and write median value in the title of each axes
    fig, ax = plt.subplots(figsize=(18, 10), nrows=len(df_tmp.columns)//2, ncols=2)
    if not (isinstance(ax,list) or isinstance(ax,np.ndarray)):
        ax = [ax]
    for i, col in enumerate(df_tmp.columns):
        try:
            ax[i//2,i%2].hist(df_tmp[col], bins=int(np.sqrt(len(df_clean.index))))
            ax[i//2,i%2].set_title(f'{col}. Median = {df_tmp[col].median():.2f}')
            ax[i//2,i%2].grid()
        except Exception as e:
            print(f"Error with column {col} {e}")
    plt.tight_layout()
    return fig, ax

def plot_kaplan_meier(df_pu: pd.DataFrame, column_name: str,
                           status_column: str = "Status", survival_in_days: str = "Survival_in_days",
                           legend_dict = None):

        diff_values = sorted(df_pu[column_name].dropna().unique().tolist())

        fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1, sharex=True)
        if not isinstance(ax, np.ndarray):
            ax = [ax]
        i = 0
        kmfs = []
        p_values = {}
        at_risk_lables = []
        for s in diff_values:
            mask_treat = df_pu[column_name] == s
            p_values[s] = logrank_test(df_pu[status_column][mask_treat], df_pu[status_column][~mask_treat],
                                       df_pu[survival_in_days][mask_treat],
                                       df_pu[survival_in_days][~mask_treat]).p_value
            i += 1
            ix = df_pu[column_name] == s
            kmf = KaplanMeierFitter()
            #TODO find more general way to create lables
            full_label = ''
            if column_name.startswith('gene_'):
                label_str = 'mutated' if s == 1 else 'wildtype'
                gene_name = column_name.replace('gene_','')
                full_label = label_str + " " +gene_name
            else:
                if legend_dict is None or column_name not in legend_dict:
                    label_str = str(s)
                    full_label = column_name + " = " + label_str
                else:
                    if str(s) in legend_dict[column_name]:
                        full_label = legend_dict[column_name][str(s)]
                    elif 'legend name' in legend_dict[column_name]:
                        full_label = legend_dict[column_name]['legend name'] + " = " + str(s)

            kmf.fit(df_pu[survival_in_days][ix], df_pu[status_column][ix],
                    label=full_label + f" p-value = {p_values[s]:.5f} ")
            kmf.plot_survival_function(ax=ax[0], ci_legend=True)
            at_risk_lables.append(f"{full_label}")
            kmfs.append(kmf)
        add_at_risk_counts(*kmfs, labels=at_risk_lables, ax=ax[0])
        ax[0].set_ylabel("est. probability of survival $\hat{S}(t)$")
        ax[0].set_xlabel(f"time $t$ (days)")
        ax[0].set_title(f"Kaplan-Meier survival estimates [{survival_in_days}] ")
        plt.tight_layout()
        return fig

def keep_only_specific_columns(df, keep_columns, ignore_columns):
    return df.loc[:, [col for col in df.columns if
                      col in keep_columns or
                      col not in ignore_columns]]


from version import __version__
if __name__ == '__main__':
    list_of_plot_types = ["kaplan_meier", "pieplots", "floathistograms", "valuecounts",'fisher_exact_test','kruskal_wallis_test']
    parser = argparse.ArgumentParser(description=f"Plot figures for survival analysis (ver: {__version__})",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_csv", help="Input CSV file", type=str, required=True)
    parser.add_argument("--output_pdf", help="Output file with figures", type=str, required=True)
    parser.add_argument("--plot", help="Type of plot", choices=list_of_plot_types, default="kaplan_meier")
    parser.add_argument("--status_col", help="Column with status (event occur or not) ", type=str, default="status")
    parser.add_argument("--survival_time_col", help="Time until event ", type=str, default="survival_in_days")
    parser.add_argument("--patient_id_col", help="Patients id", type=str, default="patient_id")
    parser.add_argument("--columns", help="One or few columns for plot", type=str, default="")
    parser.add_argument("--min_size_of_group", help="Minimal group for Kaplan-Meier plots as fraction of all casses", type=float, default=0.07)
    parser.add_argument("--max_amount_of_groups", help="Maximum number of groups per factor", type=str,
                        default=10)
    parser.add_argument("--max_survival_length", help="Maximum consider time interval in Kaplan-meier plots", type=float,
                        default=365*5)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')
    parser.add_argument("--verbose", help="Verbose mode", type=int, default=1)
    parser.add_argument('--custom_legend', help="Path to json file with custom legends for KM plots", type=str, default=None)
    parser.add_argument("--filter_nan_columns", help="comma separated list of columns where NaN will be detected and filetered", default="")
    parser.add_argument("--title", help="Title of plot", type=str, default="")
    parser.add_argument("--tiff", help="If set, plots will be saved in tiff format", default=False, action='store_true')
    args = parser.parse_args()
    input_csv = args.input_csv

    status_col = args.status_col
    tiff_dpi = 100
    survival_time_col = args.survival_time_col
    patient_id_col = args.patient_id_col
    show = args.show
    plot_type = args.plot
    if args.custom_legend is not None:
        with open(args.custom_legend, 'r') as fid:
            legend_dict = json.load(fid)
    else:
        legend_dict = None
    df = pd.read_csv(input_csv, delimiter=',')
    df.reset_index(inplace=True)

    if args.filter_nan_columns != "":
        columns_to_filter = args.filter_nan_columns.split(',')
        if verbose > 1:
            print(f"survplots:Number of rows before NaN in columns {columns_to_filter} filtering:{len(df.index)}")
        df = df.dropna(subset=columns_to_filter)
        if verbose > 1:
            print(f"survplots:Number of rows after NaN in columns {columns_to_filter} filtering:{len(df.index)}")
    pp = PdfPages(args.output_pdf)
    if args.columns == "":
        columns = [col for col in df.columns if col not in [status_col, survival_time_col, patient_id_col]]
    elif args.columns.endswith('*'):
        columns = [col for col in df.columns if col.startswith(args.columns[:-1])]
    else:
        columns = args.columns.split(',')
    if args.verbose > 1:
        print(f"survplots:Columns subjected for plot and analysis: {columns}")
    #remove all kind of none values from status column
    number_of_rows_before_status_column_nan_filtering = len(df.index)
    df = df.dropna(subset=[status_col])
    if number_of_rows_before_status_column_nan_filtering != len(df.index):
        print(f"survplots:Warning: Number of rows after NaN in status column filtering:{len(df.index)}")
    #check if status column is binary
    if df[status_col].nunique() != 2:
        raise RuntimeError(f"Column {status_col} is not binary")
    #convert status column to boolean
    df[status_col] = df[status_col].astype(bool)


    if plot_type == "kaplan_meier":
        i = 0
        df.loc[df[survival_time_col] > args.max_survival_length, status_col] = False
        df.loc[df[survival_time_col] > args.max_survival_length, survival_time_col] = args.max_survival_length
        min_group_size = int(args.min_size_of_group * len(df))
        max_number_of_groups = args.max_amount_of_groups

        for col in tqdm.tqdm(columns, desc="Plotting kaplan_meier", disable=args.verbose != 1):
            #check if column is categorial
            i += 1
            if df[col].dtype == 'categorical':
                if args.verbose > 1:
                    print(f"Column {col} is not categorical, skip it")
                continue
            if df[col].nunique() > max_number_of_groups:
                if args.verbose > 1:
                    print(f"Column {col} has too many unique values, skip it")
                continue
            #compute number of cases of each value and skip group if number of cases is less than min_group_size
            df_filtered = deepcopy(df)
            continue_flag = False
            n_groups = df[col].unique()
            for j in df[col].unique():
                if j is None or pd.isna(j):
                    continue
                if df[col].value_counts()[j] < min_group_size:
                    if args.verbose > 1:
                        print(f"Column {col} has too few cases of {j} {df[col].value_counts()[j]}, remove this group")
                    df_filtered = df_filtered[df_filtered[col] != j]
                    if len(df_filtered.index) < min_group_size:
                        print(f"Column {col} has too few cases in all groups and will be skipped")
                        continue_flag = True
            if continue_flag:
                continue

            if args.verbose > 1:
                print(f"Plotting kaplan_meier for column {col} {len(columns)}\{i}. Number of unique values is {df[col].nunique()}. Number of Nulls is {df[col].isnull().sum()}")
            try:
                fig = plot_kaplan_meier(df_filtered, col, status_col, survival_time_col, legend_dict=legend_dict)
                pp.savefig(fig)
                if args.tiff:
                    fig.savefig(f"{args.output_pdf[:-4]}_{col}.tiff", dpi=tiff_dpi, format='tiff')
            except Exception as e:
                print(f"Error while plotting kaplan_meier for column {col}: {str(e)}")
                raise e
    elif plot_type == "pieplots":
        fig, ax = plot_piecharts_of_categorial_variables(df.loc[:,columns])
        pp.savefig(fig)
        if args.tiff:
            fig.savefig(f"{args.output_pdf[:-4]}_pieplots.tiff", dpi=tiff_dpi, format='tiff')
    elif plot_type == "floathistograms":
        fig, ax = plot_histograms_of_float_values(df.loc[:,columns])
        pp.savefig(fig)
        if args.tiff:
            fig.savefig(f"{args.output_pdf[:-4]}_floathistograms.tiff", dpi=tiff_dpi, format='tiff')
    elif plot_type == "valuecounts":
        fig, ax = plot_value_counts(df, columns)
        pp.savefig(fig)
        if args.tiff:
            fig.savefig(f"{args.output_pdf[:-4]}_valuecounts.tiff", dpi=tiff_dpi, format='tiff')
    elif plot_type == "fisher_exact_test":
        columns_prefix = ""
        if args.columns.endswith('*'):
            columns_prefix = args.columns[:-1]

        binary_outcome_column = args.status_col
        #check that all columns are binary
        for col in columns:
            if df[col].nunique() != 2:
                print(f"survplots:Warning! Column {col} is not binary, will skip it")
        columns_binary = [col for col in columns if df[col].nunique() == 2]
        good_outcome_factor_true = {}
        good_outcome_factor_false = {}
        bad_outcome_factor_true = {}
        bad_outcome_factor_false = {}
        tdf_good_response = df[df[binary_outcome_column] == True]
        tdf_bad_response = df[df[binary_outcome_column] == False]
        fisher_results = {}
        p_value_threshold = 0.1
        p_value_threshold2 = 0.05

        table_data = []
        raw_lables = []
        total_number_of_cases = []
        for col in columns_binary:
            total_number_of_cases.append(sum(df[col]))
        #select top 10 columns with maximal number of cases
        columns_top_10 = [x for _, x in sorted(zip(total_number_of_cases, columns_binary), reverse=True)][:10]
        columns_top_10_names = [x.replace(columns_prefix,'') for x in columns_top_10]
        if args.verbose > 1:
            print(f"Top 10 columns with maximal number of cases: {columns_top_10_names}")
        for col in columns_binary:
            good_outcome_factor_true[col] = tdf_good_response[col].sum()
            good_outcome_factor_false[col] = len(tdf_good_response) - good_outcome_factor_true[col]
            bad_outcome_factor_true[col] = tdf_bad_response[col].sum()
            bad_outcome_factor_false[col] = len(tdf_bad_response) - bad_outcome_factor_true[col]

            ftable = [[good_outcome_factor_true[col], good_outcome_factor_false[col]] ,
                      [bad_outcome_factor_true[col], bad_outcome_factor_false[col]]]
            #continue if some of the value for Fisher test are 0
            if 0 in ftable[0] or 0 in ftable[1]:
                continue
            oddsratio, pvalue = fisher_exact(ftable)
            fisher_results[col] = (oddsratio,pvalue)
            if pvalue < p_value_threshold and args.verbose > 1:
                print(f"Factor {col} oddsratio {oddsratio:.4f} pvalue {pvalue:.4f} [TP TN FP FN]:{ftable}")
                raw_lables.append(col)
                table_data.append([ftable[0][0],ftable[0][1],ftable[1][0],ftable[1][1], oddsratio,pvalue])

        #plot fisher results as scatter plot
        fig,ax = plt.subplots(figsize=(10,10))
        pvalue = [x[1] for x in fisher_results.values()]
        oddsratio = [x[0] for x in fisher_results.values()]
        #replace inf in ods ratio with 100
        oddsratio = [10 if x == float('inf') else x for x in oddsratio]
        genes = [x for x in fisher_results.keys()]
        #remove perfix genes_ from gene names
        genes = [x.replace(columns_prefix,'') for x in genes]
        ax.scatter(np.log2([x[0] for x in fisher_results.values()]),-np.log10([x[1] for x in fisher_results.values()]))
        ax.set_xlabel('Log2(Odds ratio)')
        ax.set_ylabel('-Log10(P-value)')
        ax.grid()
        #select pvalue  < 0.05 and plot them in red
        all_results = pd.DataFrame({'log2(OddsRatio)':np.log2(oddsratio),'-log10(p-value)':-np.log10(pvalue),'name':genes},index=genes)
        significant = all_results[all_results['-log10(p-value)'] > -np.log10(p_value_threshold)]
        plt.scatter(significant['log2(OddsRatio)'], significant['-log10(p-value)'], color='red')

        #TODO: implement more general and robust solution for text shifts
        txt_shift_dict = {}
        for i, txt in enumerate(significant.index):
            k = significant['log2(OddsRatio)'][i]*10 + significant['-log10(p-value)'][i]
            if k not in txt_shift_dict:
                txt_shift_dict[k] = 0
            else:
                txt_shift_dict[k] += 1
            ax.annotate("  " + txt, (significant['log2(OddsRatio)'][i], significant['-log10(p-value)'][i]-txt_shift_dict[k]*0.035),
                        rotation=0 * int(i) % 360, fontsize=8,ha='left')

        txt_shift_dict2 = {}
        #plot text labels for most frequent columns 
        for i, txt in enumerate(all_results.index):
            print(f"i={txt} txt={columns_top_10_names}")
            if txt in columns_top_10_names and txt not in significant.index.tolist():

                k = all_results['log2(OddsRatio)'][i] * 10 + all_results['-log10(p-value)'][i]
                if k not in txt_shift_dict2:
                    txt_shift_dict2[k] = 0
                else:
                    txt_shift_dict2[k] += 1
                ax.annotate("  " + txt, (
                all_results['log2(OddsRatio)'][i], all_results['-log10(p-value)'][i] - txt_shift_dict2[k] * 0.035),
                        rotation=0 * int(i) % 360, fontsize=8, ha='left')

        ax.axhline(-np.log10(p_value_threshold), color='r', linestyle='--')
        # plot text near line with p_value_threshold
        ax.text(0.1, -np.log10(p_value_threshold) + 0.02, f'p-value = {p_value_threshold}', rotation=0, fontsize=12)
        ax.axhline(-np.log10(p_value_threshold2), color='r', linestyle='-.')
        # plot text near line with p_value_threshold
        ax.text(0.1, -np.log10(p_value_threshold2) + 0.02, f'p-value = {p_value_threshold2}', rotation=0, fontsize=12)

        # and vertical line at log2(oddsratio) = 0
        ax.axvline(0, color='k', linestyle='-', linewidth=1)
        plt.title(f'{args.title} Exact Fisher test. ')
        plt.tight_layout()
        pp.savefig(fig)
        if args.tiff:
            fig.savefig(f"{args.output_pdf[:-4]}_eft.tiff", dpi=tiff_dpi, format='tiff')
    elif plot_type == "kruskal_wallis_test":
        i = 0
        for col in columns:
            i = i + 1
            fig, ax = plot_kruskal_wallis_boxplot(df, col, survival_time_col, legend_dict=legend_dict)
            pp.savefig(fig)
            if args.tiff:
                fig.savefig(f"{args.output_pdf[:-4]}_kruskal_wallis_test_{col}.tiff", dpi=tiff_dpi, format='tiff')

    if show:
        plt.show()
    pp.close()
    plt.close(fig)