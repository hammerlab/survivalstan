import os
from fnmatch import fnmatch
import ntpath
import pkg_resources
import seaborn as sb
import matplotlib.pyplot as plt
from survivalstan import extract_baseline_hazard
import pandas as pd
import numpy as np
from lifelines.utils import survival_table_from_events

def _summarize_survival(df, time_col, event_col, evaluate_at=None):
    ## prepare survival table
    table = survival_table_from_events(df[time_col], df[event_col])
    table.reset_index(inplace=True)
    ## normalize survival as fraction of initial_n
    table['initial_n'] = table.loc[table['event_at'] == 0.0,'at_risk'][0]
    table['survival'] = table.apply(lambda row: row['at_risk']/row['initial_n'], axis=1)
    ## handle timepoints if given
    if evaluate_at is not None:
        evaluate_times = pd.DataFrame({'event_at': evaluate_at})
        table = pd.merge(evaluate_times, table, on='event_at', how='outer')
        table = table.sort_values('event_at').fillna(method='ffill')
        table['keep'] = table['event_at'].apply(lambda x: x in evaluate_at)
    else:
        table['keep'] = True
    table = table.loc[table['keep'] == True,['event_at','survival']]
    table.rename(columns={'event_at': time_col}, inplace=True)
    return table


def _get_sample_ids_single_model(model, sample_col='patient_id'):
    patient_sample_ids = model['df'].loc[:,[sample_col,'sample_id']].drop_duplicates().sort_values('sample_id')
    patient_sample_ids['model_cohort'] = model['model_cohort']
    return patient_sample_ids


def get_sample_ids(models, sample_col='patient_id'):
    data = [_get_sample_ids_single_model(model=model, sample_col=sample_col) for model in models]
    return pd.concat(data)


def _prep_yhat_data_single_model(model, time_element='y_hat_time', event_element='y_hat_event',
                                 time_col='event_time', event_col='event_status', sample_col='patient_id',
                                 varnames=None):
    pp_event_time = extract_params_long(models=[model],
                                        element=time_element,
                                        varnames=varnames,
                                       )
    pp_event_time.rename(columns={'value': time_col, 'variable': sample_col}, inplace=True)
    pp_event_status = extract_params_long(models=[model],
                                          element=event_element,
                                          varnames=varnames,
                                         )
    pp_event_status.rename(columns={'value': event_col, 'variable': sample_col}, inplace=True)
    pp_data = pd.merge(pp_event_time, pp_event_status, on=['iter', sample_col, 'model_cohort'])
    return pp_data


def _prep_pp_data_single_model(model, time_element='y_hat_time', event_element='y_hat_event',
                               sample_col='patient_id', use_sample_id=True,
                               time_col='event_time', event_col='event_status'):
    if use_sample_id:
        patient_sample_ids = _get_sample_ids_single_model(model=model, sample_col=sample_col)
        varnames = patient_sample_ids[sample_col].values
    else:
        varnames = None
    pp_data = _prep_yhat_data_single_model(model=model,
                                           event_element=event_element, 
                                           time_element=time_element,
                                           event_col=event_col,
                                           time_col=time_col,
                                           sample_col=sample_col,
                                           varnames=varnames)
    return pp_data


def prep_pp_data(models, time_element='y_hat_time', event_element='y_hat_event',
                 sample_col='patient_id', event_col='event_status', time_col='event_time',
                 use_sample_id=True):
    data = [_prep_pp_data_single_model(model=model, sample_col=sample_col, event_element=event_element, 
                                       time_element=time_element, event_col=event_col, time_col=time_col,
                                       use_sample_id=use_sample_id)
            for model in models]
    data = pd.concat(data)
    data.sort_values([time_col,'iter'], inplace=True)
    return data


def prep_pp_survival_data(models, time_element='y_hat_time', event_element='y_hat_event',
                          sample_col='patient_id', time_col='event_time', event_col='event_status'):
    pp_data = prep_pp_data(models, time_element=time_element, event_element=event_element,
                           sample_col=sample_col, time_col=time_col, event_col=event_col)
    pp_surv = pp_data.groupby(['iter','model_cohort']).apply(
         lambda df: _summarize_survival(df, time_col=time_col, event_col=event_col))
    return pp_surv


def prep_oos_survival_data(models, time_element='y_oos_time', event_element='y_oos_event',
                          time_col='event_time', event_col='event_status', sample_col='sample_id'):
    oos_data = prep_pp_data(models, time_element=time_element, event_element=event_element,
                            sample_col=sample_col, time_col=time_col, event_col=event_col,
                            use_sample_id=False)
    oos_surv = oos_data.groupby(['iter','model_cohort']).apply(
         lambda df: _summarize_survival(df, time_col=time_col, event_col=event_col))
    return oos_surv


def _plot_pp_survival_data(pp_surv, time_col='event_time', survival_col='survival',
                           num_ticks=10, step_size=None, ticks_at=None, **kwargs):
    pp_surv.sort_values(time_col, inplace=True)
    f, ax = plt.subplots(1, 1)
    if ticks_at is None:
        x_min = min(pp_surv[time_col].drop_duplicates())
        x_max = max(pp_surv[time_col].drop_duplicates())
        if step_size is None:
            step_size = (x_max - x_min)/num_ticks
        ticks_at = np.arange(start=x_min, stop=x_max, step=step_size)
    survival_plot = pp_surv.boxplot(
        column=survival_col,
        by=time_col,
        whis=[2.5, 97.5],
        positions=pp_surv[time_col].drop_duplicates(),
        ax=ax,
    )
    f.suptitle('')
    _ = plt.ylim([0, 1])
    _ = plt.xticks(rotation="vertical")
    _ = plt.xlabel('Days')
    _ = plt.ylabel('Survival %')
    _ = plt.title('')
    _ = ax.xaxis.set_ticks(ticks_at)
    _ = ax.xaxis.set_ticklabels(
         [r"%d" % (int(round(x))) for x in ticks_at])

    if dict(**kwargs):
        _ = plt.setp(survival_plot[survival_col]['boxes'], **kwargs)
        _ = plt.setp(survival_plot[survival_col]['whiskers'], **kwargs)

def plot_pp_survival(models, time_element='y_hat_time', event_element='y_hat_event', sample_col='patient_id',
                     num_ticks=10, step_size=None, ticks_at=None, time_col='event_time', event_col='event_status', **kwargs):
    pp_surv = prep_pp_survival_data(models, time_element=time_element, event_element=event_element,
                                    sample_col=sample_col, time_col=time_col, event_col=event_col)
    _plot_pp_survival_data(pp_surv, num_ticks=num_ticks, step_size=step_size, ticks_at=ticks_at, time_col=time_col, **kwargs)


def plot_observed_survival(df, event_col, time_col, *args, **kwargs):
    actual_surv = _summarize_survival(df=df, time_col=time_col, event_col=event_col)
    plt.plot(actual_surv[time_col], actual_surv['survival'], label='observed', *args, **kwargs)

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


def plot_coefs(models, element='coefs', force_direction=None, trans=None):
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
    trans (function, optional):
        If present, transforms value of `value` column
            - example: np.exp to plot exp(beta)
        if None (default), plots raw value

    """

    # TODO: check if models object is a list or a single model

    # prep data from models given
    if element=='baseline' or element=='baseline_raw':
        value, variable, df = _prep_data_for_baseline_hazard(models, element=element)
    else:
        value, variable, df = _prep_data_for_coefs(models=models, element=element)
    
    if trans:
        df[value] = trans(df[value])

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
    if varnames is None:
        df = pd.DataFrame(
            model['fit'].extract()[element]
        )
    else: 
        df = pd.DataFrame(
            model['fit'].extract()[element]
            , columns=varnames
        )
    if rename_vars is not None:
        df.rename(columns = rename_vars, inplace=True)
    df.reset_index(0, inplace = True)
    df = df.rename(columns = {'index':'iter'})
    df = pd.melt(df, id_vars = ['iter'])
    df['model_cohort'] = model['model_cohort']
    return(df)

