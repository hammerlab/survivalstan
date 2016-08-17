import os
from fnmatch import fnmatch
import ntpath
import pkg_resources
import seaborn as sb
import matplotlib.pyplot as plt
from survivalstan import extract_baseline_hazard
import pandas as pd
import numpy as np

def _list_files_in_path(path, pattern="*.stan"):
    """
    indexes a directory of stan files
    returns as dictionary containing contents of files
    """

    results = []
    for dirname, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                results.append(os.path.join(dirname, name))
    return(results)


def _read_file(filepath, resource=None):
    """
    reads file contents from disk

    Parameters
    ----------
    filepath (string):
        path to file (can be relative or absolute)
    resource (string, optional):
        if given, path is relative to package install root
        used to load stan files provided by packages
        (e.g. those within a package library)

    Returns
    -------
    The specifics of the return type depend on the value of `resource`.
        - if resource is None, returns contents of file as a character string
        - otherwise, returns a "resource_string" which
            acts as a character string but technically isn't one.
    """
    print(filepath)
    if not(resource):
        with open(filepath, 'r') as myfile:
            data = myfile.read()
    else:
        data = pkg_resources.resource_string(
            resource, filepath)
    return data


def read_files(path, pattern='*.stan', encoding="utf-8", resource=None):
    """
    Reads file contents from a directory path into memory. Returns a 
    dictionary of file names: file contents.

    Is intended to be used to load a directory of stan files into an object.

    Parameters
    ----------
    path (string):
        directory path (can be relative or absolute)
    pattern (string, optional):
        regex pattern applied to files on import
        defaults to "*.stan"
    encoding (string, optional):
        encoding to use when importing files
        defaults to "UTF-8"
    resource (string, optional):
        if given, path is relative to package install root
        used to load stan files provided by packages
        (e.g. those within a package library)

    Returns
    -------
    The specifics of the return type depend on the value of `resource`.
        - if resource is None, returns contents of file as a character string
        - otherwise, returns a "resource_string" which
            acts as a character string but technically isn't one.
    """
    files = _list_files_in_path(path=path, pattern=pattern)
    results = {}
    for file in files:
        file_data = {}
        file_data['path'] = file
        file_data['basename'] = ntpath.basename(file)
        file_data['code'] = _read_file(
            file,
            resource=resource).decode(encoding)
        results[file_data['basename']] = file_data['code']
    return(results)

def _prep_data_for_coefs(models, element):
    """ 
    Helper function to concatenate/extract data 
    from a list of model objects.

    See `plot_coefs` for description of data inputs
    """

    # concatenate data from models given
    df_list = list()
    [df_list.append(model[element]) for model in models]
    df = pd.concat(df_list, ignore_index=True)
    return 'value', 'variable', df


def _prep_data_for_baseline_hazard(models, element='baseline'):
    """ 
    Helper function to concatenate/extract baseline hazard data 
    from a list of model objects.
    
    Note `element` input parameter is ignored here.

    See `plot_coefs` for description of data inputs
    """
    # prepare df containing posterior estimates of baseline hazards
    df_list = list()
    [df_list.append(extract_baseline_hazard(model, element=element)) for model in models]
    df = pd.concat(df_list)

    # add helper variables to df
    df['timepoint_id'] = df['timepoint_id'].astype('category')
    df['log_hazard'] = np.log1p(df['baseline_hazard'])
    df['end_time_id'] = df['end_time'].astype('category')
    return 'log_hazard', 'end_time_id', df


def plot_coefs(models, element='coefs', force_direction=None):
    """
    Plot coefficients for models listed

    Parameters
    ----------

    models (list):
        List of model objects
    element (string, optional):
        Which element to plot. defaults to 'coefs'.
        Other options (depending on model type) include: 
        - 'grp_coefs'
        - 'baseline'
    force_direction (string, optional):
        Takes values 'h' or 'v'
            - if 'h': forces horizontal orientation, (`variable` names along the x axis)
            - if 'v': forces vertical orientation (`variable` names along the y axis)
        if None (default), coef plots default to 'v' for all plots except baseline hazard.

    """

    # TODO: check if models object is a list or a single model

    # prep data from models given
    if element=='baseline' or element=='baseline_raw':
        value, variable, df = _prep_data_for_baseline_hazard(models, element=element)
    else:
        value, variable, df = _prep_data_for_coefs(models=models, element=element)


    # select hue depending on number of elements
    if len(models)==1:
        hue = None
    else:
        hue = 'model_cohort'

    if element=='baseline' or element=='baseline_raw':
        direction = 'h'
    else:
        direction = 'v'

    if force_direction:
        direction = force_direction

    if direction=='h':
        xval = variable
        yval = value
    else:
        xval = value
        yval = variable

    ## plot coefficients
    sb.boxplot(x = xval, y = yval, data = df, hue = hue)
    if hue=='model_cohort':
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def extract_params_long(models, element, rename_vars=None, varnames=None):
    """
    Helper function to extract & reformat params

    Parameters
    ----------

    models (list):
        List of model objects
    element (string, optional):
        Which element to plot. defaults to 'coefs'.
        Other options (depending on model type) include: 
        - 'grp_coefs'
        - 'baseline_hazard'
    rename_vars (dict, optional):
        - dictionary mapping from integer positions (0, 1, 2) to variable names
    varnames (list of strings, optional):
        - list of variable names to apply to columns from the extracted object

    Returns
    -------
    Pandas dataframe containing posterior draws per iteration

    """
    df_list = list()
    for model in models:
        df_list.append(_extract_params_from_single_model(
            model,
            element = element,
            rename_vars=rename_vars,
            varnames=varnames
            ))
    df_list = pd.concat(df_list)
    return(df_list)


def _extract_params_from_single_model(model, element, rename_vars=None, varnames=None):
    if not varnames:
        df = pd.DataFrame(
            model['fit'].extract()[element]
        )
    else: 
        df = pd.DataFrame(
            model['fit'].extract()[element]
            , columns=varnames
        )
    if rename_vars:
        df.rename(columns = rename_vars, inplace=True)
    df.reset_index(0, inplace = True)
    df = df.rename(columns = {'index':'iter'})
    df = pd.melt(df, id_vars = ['iter'])
    df['model_cohort'] = model['model_cohort']
    return(df)

