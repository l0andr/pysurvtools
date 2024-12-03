import os.path

import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import warnings
import argparse
import json
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import dtreeviz


from tqdm import tqdm
import pickle

class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()

def categorical_variable_renaming(df_formodel: pd.DataFrame, verbose=0,legend_dict:dict={},ncat_treshold=5):
    for column in df_formodel.columns:
        if column in legend_dict and 'legend name' in legend_dict[column]:
            if verbose >= 1:
                print(f"Column {column} will be renamed to {legend_dict[column]['legend name']}")
            df_formodel.rename(columns={column: legend_dict[column]['legend name']}, inplace=True)
            column = legend_dict[column]['legend name']
        if len(df_formodel[column].unique().tolist()) <= ncat_treshold and not df_formodel[column].dtype == bool:
            if column in legend_dict:
                #for each unique value in column, rename it to value from legend_dict
                for val in df_formodel[column].unique().tolist():
                    if val is None or val == np.nan or val == 'nan' or pd.isna(val):
                        continue
                    if str(val) in legend_dict[column]:
                        if verbose >= 1:
                            print(f"\t In column {column} rename {val} to {legend_dict[column][str(val)]} ")
                        df_formodel[column] = df_formodel[column].replace(val,legend_dict[column][str(val)])
                    else:
                        if verbose >= 3:
                            print(f"\t In column {column} value {val} not found in legend_dict")

    return df_formodel

def categorical_parameters_to_bool(df_formodel: pd.DataFrame, except_columns: list = None,verbose=0,ncat_treshold=5):
    column_for_dropping = []
    factor_columns = []
    for column in (set(df_formodel.columns) - {except_columns}):
        if len(df_formodel[column].unique().tolist()) == 1:
            if verbose >= 1:
                print(f"{column} will be dropped (variance = 0)")
            column_for_dropping.append(column)
        if df_formodel[column].dtype == int and sorted(df_formodel[column].unique().tolist()) == [0, 1]:
            if verbose >= 1:
                print(f"{column} will be converted to boolean")
            df_formodel[column] = df_formodel[column].astype('bool')
        elif len(df_formodel[column].unique().tolist()) <= ncat_treshold and not df_formodel[column].dtype == bool:
            if verbose >= 1:
                print(f"{column} will be converted to multiple columns {df_formodel[column].unique().tolist()}")
            for val in df_formodel[column].unique().tolist():
                if val is None or val == np.nan or val == 'nan' or pd.isna(val):
                    continue
                if verbose >= 1:
                    print(f"\tBinnary column {column}={val} created")
                df_formodel[f"{column}={val}"] = df_formodel[column] == val
                df_formodel[f'{column}={val}'] = df_formodel[f'{column}={val}'].astype('bool')
            column_for_dropping.append(column)
        elif (df_formodel[column].dtype == int or df_formodel[column].dtype == np.float64 or df_formodel[
            column].dtype == np.int64) and \
                len(df_formodel[column].unique().tolist()) > 5:
            df_formodel[column] = df_formodel[column].fillna(df_formodel[column].median()).astype(np.float64)
    df_formodel.drop(columns=column_for_dropping, inplace=True)
    return df_formodel


def calculate_leaf_purity(model, X, data_y):
    # Get the tree structure
    tree = model.tree_

    # Initialize a list to store purity values
    purity_values = []
    class_counts_values = []
    # Iterate over all leaves
    for leaf_index in range(tree.node_count):
        # Check if the node is a leaf
        if tree.children_left[leaf_index] == tree.children_right[leaf_index]:  # Leaf node condition
            # Get the indices of the samples that reach this leaf
            samples_indices = np.where(model.apply(X) == leaf_index)[0]
            if len(samples_indices) > 0:
                # Get the class labels for these samples
                if isinstance(data_y, np.ndarray):
                    labels = data_y[samples_indices]  # For NumPy array
                else:
                    labels = data_y.iloc[samples_indices].values

                # Calculate purity (Gini impurity)
                class_counts = np.bincount(labels)
                probabilities = class_counts / len(labels)
                gini_impurity = 1 - np.sum(probabilities ** 2)

                # Append the purity value
                purity_values.append(gini_impurity)
                class_counts_values.append(class_counts)
            else:
                # If no samples reach this leaf, append None or a default value
                purity_values.append(None)
                class_counts_values.append(None)

    return purity_values,class_counts_values

def decision_tree_hyper_parameter_grid_search(X, data_y, min_weight_fraction_leaf, ccp_alpha, min_samples_leaf, max_depth,
                                         criterion='entropy',optimized_parameter="min_weight_fraction_leaf",min_value=0.01,max_value=0.1,npoint=10):
    td_acc = []
    cv_acc = []
    pscale = np.linspace(min_value,max_value,npoint)
    for p in pscale:
        if optimized_parameter == "min_weight_fraction_leaf":
            min_weight_fraction_leaf = p
        if optimized_parameter == "ccp_alpha":
            ccp_alpha = p
        if optimized_parameter == "min_samples_leaf":
            min_samples_leaf = int(p)
        if optimized_parameter == "max_depth":
            max_depth = int(p)
        model = DecisionTreeClassifier(min_weight_fraction_leaf=min_weight_fraction_leaf, ccp_alpha=ccp_alpha,
                                                   min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                                                   criterion=criterion,
                                                   random_state=2).fit(X, data_y)
        td_acc.append(model.score(X, data_y))
        cv_acc.append(cross_val_score(model, X, data_y, cv=5).mean())
    return td_acc, cv_acc,pscale



if __name__ == '__main__':

    list_of_plot_types = ["simple", "simple_full", "dtreeviz"]
    parser = argparse.ArgumentParser(
        description="Transform data from initial csv to csv suitable for survival analysis",
        formatter_class=argparse.RawTextHelpFormatter)
    #parameters related to input data and preprocessing of input data
    parser.add_argument("-input_csv", help="Input CSV files", type=str, required=True)
    parser.add_argument("--input_delimiter", help="Delimiter for input file", type=str, default=",")
    parser.add_argument("-ycolumn", help="Feature that should be predicted", type=str, required=True)
    parser.add_argument("--xcolumns", help="Features that should be used for split branch", type=str, default="")
    parser.add_argument("--sort_columns", help="Columns for pre-sort data before processing", type=str, default="")
    parser.add_argument("--unique_record", help="List of columns to identify unique records", type=str, default="")
    parser.add_argument("--model", help="File containing model, if set only plots will be created", type=str, default="")
    parser.add_argument("--random_seed", help="Random seed for model", type=int, default=None)
    #parameters related to output
    parser.add_argument("--verbose", help="Verbose level", type=int, default=2)
    parser.add_argument("--show", help="If set, plots will be shown", default=False,
                        action='store_true')
    parser.add_argument("--class_names", help="List of class names", type=str, default="")
    parser.add_argument("--output_model", help="File for output model", type=str, default="")
    parser.add_argument('--custom_legend', help="Path to json file with custom legends for plots", type=str,
                        default=None)
    #parameters related to predefined parameters of tree
    parser.add_argument("--min_weight_fraction_leaf", help="The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node", type=float, default=None)
    parser.add_argument("--min_samples_leaf", help="The minimum number of samples required to be at a leaf node", type=int, default=None)
    parser.add_argument("--max_depth", help="The maximum depth of the tree", type=int, default=None)
    parser.add_argument("--ccp_alpha", help="Complexity parameter used for Minimal Cost-Complexity Pruning", type=float, default=None)
    parser.add_argument("--min_impurity_decrease", help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value", type=float, default=None)
    parser.add_argument("--max_features", help="The number of features to consider when looking for the best split", type=int, default=None)
    parser.add_argument("--max_leaf_nodes", help="Grow a tree with max_leaf_nodes in best-first fashion", type=int, default=None)
    parser.add_argument("--min_samples_split", help="The minimum number of samples required to split an internal node", type=int, default=None)
    parser.add_argument("--criteria", help="The function to measure the quality of a split", type=str,
                        choices=['entropy', 'gini'], default='gini')
    parser.add_argument("--plot_type", help="Type of plot", choices=list_of_plot_types, default="simple")
    parser.add_argument("--tiff", help="tiff file name for output plot", choices=list_of_plot_types, default="")
    #parameters related to hyperparameter optimization
    parser.add_argument("--steps_of_optimization", help="Number of steps for optimization", type=int, default=20)
    parser.add_argument("--filter_nan_columns", help="comma separated list of columns where NaN will be detected and filetered", default="")

    args = parser.parse_args()
    plot_type = args.plot_type
    legend_dict = {}
    if args.custom_legend is not None:
        with open(args.custom_legend, 'r') as f:
            legend_dict = json.load(f)

    verbose = args.verbose
    input_csv = args.input_csv
    input_delimiter = args.input_delimiter
    warnings.simplefilter(action='ignore', category=FutureWarning)
    tdf_tree = pd.read_csv(input_csv, delimiter=input_delimiter)
    if args.filter_nan_columns != "":
        columns_to_filter = args.filter_nan_columns.split(',')
        if args.verbose > 1:
            print(f"adaptree:Number of rows before NaN in columns {columns_to_filter} filtering:{len(tdf_tree.index)}")
        tdf_tree = tdf_tree.dropna(subset=columns_to_filter)
        if args.verbose > 1:
            print(f"adaptree:Number of rows after NaN in columns {columns_to_filter} filtering:{len(tdf_tree.index)}")

    if args.ycolumn not in tdf_tree.columns:
        raise Exception(f"adaptree:Column {args.ycolumn} not found in input CSV file")


    if args.class_names != "":
        class_names = args.class_names.split(',')
        if len(class_names) != tdf_tree[args.ycolumn].nunique():
            raise Exception(f"adaptree:Number of classes in class_names is not equal to number of unique values in {args.ycolumn}")
    if args.sort_columns:
        tdf_tree.sort_values(by=args.sort_columns.split(','), inplace=True)
    if args.unique_record:
        tdf_tree = tdf_tree.drop_duplicates(subset=args.unique_record.split(','), keep='first')

    if args.xcolumns:
        n = 0
        for col in args.xcolumns.split(','):
            if col not in tdf_tree.columns:
                if verbose > 0:
                    print(f"adaptree:Warning:Column {col} not found in input CSV file")
                continue
            n = n + 1
        if n == 0:
            raise Exception(f"adaptree:No specified data columns found in input CSV file. May be delimiter incorrect?")
        tdf_tree = tdf_tree[args.xcolumns.split(',') + [args.ycolumn]]
    else:
        tdf_tree = tdf_tree[[x for x in tdf_tree.columns if x != args.ycolumn]]


    criterium = args.criteria
    tdf_tree = categorical_variable_renaming(df_formodel=tdf_tree, verbose=3, legend_dict=legend_dict)
    tdf_tree = categorical_parameters_to_bool(df_formodel=tdf_tree, except_columns=args.ycolumn, verbose = args.verbose)

    for col in tdf_tree.columns:
        try:
            if tdf_tree[col].nunique() == 2:
                list_of_values = sorted(tdf_tree[col].unique(), reverse=True)
                if args.verbose > 0:
                    print(f"adaptree: Column {col} has 2 unique values, True is  {list_of_values[0]}")
                #check if str is in list of values
                tdf_tree[col] =tdf_tree[col].apply(lambda x: x == list_of_values[0])
                tdf_tree[col] = tdf_tree[col].astype(int)
            #if column not numeric, convert it to integer renumber the values
            if tdf_tree[col].dtype == 'object':
                # print unique values and cat codes
                if args.verbose > 0:
                    print(f"Column {col} has {tdf_tree[col].nunique()} unique values")
                    print(tdf_tree[col].value_counts())
                tdf_tree[col] = tdf_tree[col].astype('category').cat.codes
            #if column has missing values, fill them with median
            if tdf_tree[col].isnull().sum() > 0:
                if args.verbose > 0:
                    print(f"Column {col} has null values ({tdf_tree[col].isnull().sum()}) from {len(tdf_tree.index)}")
                tdf_tree[col] = tdf_tree[col].fillna(tdf_tree[col].median())
        except Exception as e:
            import traceback
            print(f"Error in column {col}: {e}. Column will be dropped")
            traceback.print_exc()
            tdf_tree.drop(columns=[col],inplace=True)


    data_y = tdf_tree[args.ycolumn].astype(int)
    X = tdf_tree.drop(columns=[args.ycolumn])
    #sort X by columns names
    X = X.reindex(sorted(X.columns), axis=1)
    model_file = args.model
    if not os.path.exists(model_file):
        if verbose > 0:
            print(f"Model file {model_file} not found - attempt of training new model")
        model_file = ""
    if model_file == "":
        params_dict = {'max_depth':args.max_depth,
         'min_samples_leaf':args.min_samples_leaf,
         'min_weight_fraction_leaf':args.min_weight_fraction_leaf,
         'ccp_alpha':args.ccp_alpha,
         'min_impurity_decrease':args.min_impurity_decrease,
         'max_features':args.max_features,
         'max_leaf_nodes':args.max_leaf_nodes,
         'min_samples_split':args.min_samples_split,
         'random_state':args.random_seed,
         'splitter':'best'}

        params_dict_max = {'max_depth': 15,
                       'min_samples_leaf': 30,
                       'min_weight_fraction_leaf': 0.1,
                       'ccp_alpha': 0.1,
                       'min_impurity_decrease': 0.1,
                       'max_features': 40,
                       'max_leaf_nodes': 40,
                       'min_samples_split': 30,
                       'random_state': 10}

        list_of_int_parameters = ['max_depth','min_samples_leaf','max_features','max_leaf_nodes','min_samples_split','random_state']
        list_of_real_parameters = ['min_weight_fraction_leaf','ccp_alpha','min_impurity_decrease']
        space = []
        list_of_parameters_for_optimization = []
        dictionary_of_fixed_parameters = {}
        for key,value in params_dict.items():
            if value is None:
                if key in list_of_int_parameters:
                    v = params_dict_max[key]
                    space.append(Integer(2, v, name=key))
                    if args.verbose > 0:
                        print(f"Parameter {key} subjected for optimization in range 2-{v}")
                elif key in list_of_real_parameters:
                    v = params_dict_max[key]
                    space.append(Real(0.0001, v, name=key))
                    if args.verbose > 0:
                        print(f"Parameter {key} subjected for optimization in range 0.0001-0.5")
                else:
                    raise Exception(f"Unknown parameter type for {key}")
                list_of_parameters_for_optimization.append(key)
            else:
                dictionary_of_fixed_parameters[key] = value

        @use_named_args(space)
        def objective1(**params):
            model = DecisionTreeClassifier(**params,**dictionary_of_fixed_parameters,
                                           criterion=criterium).fit(X, data_y)

            return -model.score(X, data_y)

        @use_named_args(space)
        def objective_mean(**params):
            ms = 0
            for n in range(0, 11):
                model = DecisionTreeClassifier(**params, **dictionary_of_fixed_parameters,
                                               criterion=criterium, random_state=n).fit(X, data_y)
                ms = ms + model.score(X, data_y)
            ms = ms / 11 * -1
            return ms

        @use_named_args(space)
        def objective_max(**params):
            ms = 0
            for n in range(0, 11):
                model = DecisionTreeClassifier(**params, **dictionary_of_fixed_parameters,
                                               criterion=criterium, random_state=n).fit(X, data_y)
                if model.score(X, data_y) > ms:
                    ms = model.score(X, data_y)
            return -ms

        @use_named_args(space)
        def objective_max_mean_purity(**params):
            model = DecisionTreeClassifier(**params, **dictionary_of_fixed_parameters,
                                               criterion=criterium).fit(X, data_y)
            purity,cc = calculate_leaf_purity(model, X, data_y)
            return sum(purity)/len(purity) #-model.score(X, data_y)/5

        @use_named_args(space)
        def objective_max_weighted_purity(**params):
            model = DecisionTreeClassifier(**params, **dictionary_of_fixed_parameters,
                                               criterion=criterium).fit(X, data_y)
            purity,cc = calculate_leaf_purity(model, X, data_y)
            for i in range(0,len(purity)):
                purity[i] = purity[i]**2 * sum(cc[i])
            return sum(purity)




        optimization_verbose = False
        if verbose > 1:
            optimization_verbose = True
        callback = []
        if verbose == 1:
            callback = [tqdm_skopt(total=args.steps_of_optimization, desc="Gaussian Process")]
        res_gp = gp_minimize(objective_max_mean_purity, space, n_calls=args.steps_of_optimization, verbose = optimization_verbose, random_state=0,n_jobs=-1,
                             callback=callback)

        print("Results of optimization:")
        dictionary_of_optimised_parameters = {}
        for i in range(0,len(list_of_parameters_for_optimization)):
            if args.verbose > 1:
                print(f"{list_of_parameters_for_optimization[i]}:{res_gp.x[i]}")
            dictionary_of_optimised_parameters[list_of_parameters_for_optimization[i]] = res_gp.x[i]
        from skopt.plots import plot_convergence
        joint_params_dict = {**dictionary_of_optimised_parameters,**dictionary_of_fixed_parameters}
        random_seed = args.random_seed
        if verbose >= 1:
            print(f"Optimized parameters:{joint_params_dict}")
        if random_seed is None:
            if 'random_state' in joint_params_dict:
                random_seed = joint_params_dict['random_state']
            else:
                random_seed = 0
            #exclude random_state from joint_params_dict
        if 'random_state' in joint_params_dict:
            if verbose > 0:
                print(f"Optimized random seed was set to {random_seed}")
            joint_params_dict.pop('random_state')
        model = DecisionTreeClassifier(**joint_params_dict,
                                       criterion=criterium,random_state=random_seed).fit(X, data_y)

        #permutation importance
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, data_y,n_repeats = 30,random_state = 0)
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"{X.columns[i]:<8} "
                      f"{r.importances_mean[i]:.3f}"
                      f" +/- {r.importances_std[i]:.3f}")
        predictions = model.predict(X)
        if verbose > 1:
            print(f"In-sample accuracy is {np.sum(predictions == data_y) / data_y.shape[0]}")
        warnings.resetwarnings()
        if args.output_model == "":
            output_name = f"dtmodel_{joint_params_dict['ccp_alpha']:.4f}_{joint_params_dict['min_samples_leaf']}_{joint_params_dict['max_depth']}_{joint_params_dict['min_weight_fraction_leaf']:.2f}.model"
        else:
            output_name = args.output_model
        class_names = args.class_names.split(',')
        if len(class_names) == 0:
            class_names = None

        #plt.figure(figsize=(20, 10))
        model = DecisionTreeClassifier(**joint_params_dict,
                                       criterion=criterium,random_state=args.random_seed).fit(X, data_y)
        #save model using pickle
        plt.figure(figsize=(20, 10))
        plot_convergence(res_gp)
        with open(output_name, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        #load loin of parameters from model
        joint_params_dict = {}
        for key in ['ccp_alpha','min_samples_leaf','max_depth','min_weight_fraction_leaf','max_features','max_leaf_nodes','min_samples_split']:
            joint_params_dict[key] = model.get_params()[key]
        random_seed = args.random_seed
        if random_seed is None:
            if 'random_state' in joint_params_dict:
                random_seed = joint_params_dict['random_state']
            else:
                random_seed = 0
            # exclude random_state from joint_params_dict
            if 'random_state' in joint_params_dict:
                joint_params_dict.pop('random_state')
        model = DecisionTreeClassifier(**joint_params_dict,
                                       criterion=criterium,random_state=random_seed).fit(X, data_y)
        tmp = calculate_leaf_purity(model, X, data_y)
        print(tmp)
    if verbose > 0:
        print(f"Model accuracy on train data:{model.score(X, data_y)}")
        scores = cross_val_score(model, X, data_y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    title = f"Decision tree with ccp_alpha={joint_params_dict['ccp_alpha']}, min_samples_leaf={joint_params_dict['min_samples_leaf']}, max_depth={joint_params_dict['max_depth']}, " \
            f"min_weight_fraction_leaf={joint_params_dict['min_weight_fraction_leaf']}, max_features={joint_params_dict['max_features']}"
    title += f"\nModel score on train data:{model.score(X, data_y)}\n"
    title += f"Cross validated accuracy {scores.mean():.2f}  with a standard deviation of {scores.std():.2f}"
    try:
        if plot_type == 'simple':
            plot_tree(model, feature_names=X.columns, filled=True, fontsize=10,class_names=class_names,proportion=False, label='none',impurity=False)
            plt.tight_layout()
            plt.title(title)
        elif plot_type == 'simple_full':
            plot_tree(model, feature_names=X.columns, filled=True, fontsize=10,class_names=class_names,proportion=True, label='all',impurity=True)
            plt.tight_layout()
            plt.title(title)
        elif plot_type == 'dtreeviz':
            #in all columns with name starting from "gene_", replace values 0 and 1 to wildtype and mutated

            viz_model=dtreeviz.model(model,
                                       X_train=X.values, y_train=data_y,
                                       feature_names=X.columns,
                                       target_name='response', class_names=class_names)
            v = viz_model.view(fontname='monospace',title=title,fancy=True)  # render as SVG into internal object
            v.show()
            v.save(f"dtreeviz_{joint_params_dict['ccp_alpha']:.4f}_{joint_params_dict['min_samples_leaf']}_{joint_params_dict['max_depth']}_{joint_params_dict['min_weight_fraction_leaf']:.2f}.tiff",format='tiff')

    except Exception as e:
        print(class_names)
        print(X.columns)
        print(model)
        print(f"Error during plotting tree: {e}")
        raise e

    warnings.resetwarnings()

    if args.show:
        plt.show()

