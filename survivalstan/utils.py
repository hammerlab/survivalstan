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
import logging
import re
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

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


def extract_time_betas(models, element='beta_time', value_name='beta', **kwargs):
    data = [_extract_time_betas_single_model(model, element=element, value_name=value_name, **kwargs) for model in models]
    return pd.concat(data)

def _extract_time_betas_single_model(stanmodel, element='beta_time', coefs=None,
                                     value_name='beta', timepoint_id_col=None,
                                     timepoint_end_col=None):
    if not timepoint_id_col:
        timepoint_id_col = stanmodel['timepoint_id_col']
    if not timepoint_end_col:
        timepoint_end_col = stanmodel['timepoint_end_col']
    if not timepoint_id_col or not timepoint_end_col:
        raise ValueError('timepoint_id_col and timepoint_end_col are required, but were either not given or were not set by stan model')
    time_betas = stanmodel['fit'].extract()[element]
    
    # determine coef names
    coef_names = list(stanmodel['x_names'])
    num_coefs = time_betas.shape[1]
    if len(coef_names) != num_coefs:
        raise ValueError('Num coefs does not equal number of coef names. Please report this as a bug')
    logger.debug('num_coefs set to {}'.format(num_coefs))
    
    # determine which coefs to extract
    plot_coefs = list(np.arange(num_coefs))
    if coefs is not None:
        plot_coefs = [val for val in plot_coefs if coef_names[val] in coefs]
    logger.debug('plot_coefs set to {}'.format(','.join(str(plot_coefs))))
    
    # extract time-betas for each coef
    time_data = list()
    for i in plot_coefs:
        tb_df = pd.DataFrame(time_betas[:,i,:])
        tb_df = pd.melt(tb_df, var_name=timepoint_id_col, value_name=value_name)
        tb_df['coef'] = coef_names[i]
        time_data.append(tb_df)
    time_data = pd.concat(time_data)
    timepoint_data = stanmodel['df'].loc[:,[timepoint_id_col, timepoint_end_col]].drop_duplicates()
    time_data = pd.merge(time_data, timepoint_data, on=timepoint_id_col)
    time_data['exp({})'.format(value_name)] = np.exp(time_data[value_name])
    time_data['model_cohort'] = stanmodel['model_cohort']
    return(time_data)

def _get_timepoint_cols(models, timepoint_id_col, timepoint_end_col):
    if not timepoint_id_col:
        timepoint_id_col = np.unique([model['timepoint_id_col'] for model in models])
        if len(timepoint_id_col)>1:
            ValueError('timepoint_id_col is not uniform for all models. Please reformat data and set timepoint_id_col manually')
        elif len(timepoint_id_col)==1:
            timepoint_id_col = timepoint_id_col[0]
    if not timepoint_end_col:
        timepoint_end_col = np.unique([model['timepoint_end_col'] for model in models])
        if len(timepoint_end_col)>1:
            ValueError('timepoint_end_col is not uniform for all models. Please reformat data and set timepoint_end_col manually')
        elif len(timepoint_end_col)==1:
            timepoint_end_col = timepoint_end_col[0]
    if not timepoint_id_col or not timepoint_end_col:
        raise ValueError('timepoint_id_col and timepoint_end_col are required, but were either not given or were not set by model')
    return (timepoint_id_col, timepoint_end_col)
    
def _plot_time_betas(models=None, df=None, element='beta_time',
                     coefs=None, y='exp(beta)', ylabel=None, 
                     timepoint_id_col=None, timepoint_end_col=None,
                     x='timepoint_end_col', xlabel='time', 
                     subplot=None, ticks_at=None, num_ticks=10, step_size=None,
                     fill=True, value_name='beta', ylim=None, **kwargs):
    if df is None:
        df = extract_time_betas(models=models, element=element, coefs=coefs,
                                     value_name=value_name, timepoint_id_col=timepoint_id_col,
                                     timepoint_end_col=timepoint_end_col)
        timepoint_id_col, timepoint_end_col = _get_timepoint_cols(models=models,
                                                              timepoint_id_col=timepoint_id_col,
                                                              timepoint_end_col=timepoint_end_col)
        logger.debug('timepoint_id_col set to {}'.format(timepoint_id_col))
        logger.debug('timepoint_end_col set to {}'.format(timepoint_end_col))
    else:
        if not timepoint_id_col:
            timepoint_id_col = 'timepoint_id'
    if x == 'timepoint_end_col':
        time_col = timepoint_end_col
    elif x == 'timepoint_id_col':
        time_col = timepoint_id_col
    else:
        time_col = x
    logger.debug('time_col set to {}'.format(time_col))
    if not time_col:
        raise ValueError('time_col is not defined - specify name of column using `x`')
    
    if not ylabel:
        if not coefs or len(coefs)>1:
            ylabel = '{}'.format(y)
        else:
            ylabel = '{} for {}'.format(y, coefs[0])
    df.sort_values(time_col, inplace=True)
    if not subplot:
        f, ax = plt.subplots(1, 1)
    else:
        f, ax = subplot
    if ticks_at is None:
        x_min = min(df[time_col].drop_duplicates())
        x_max = max(df[time_col].drop_duplicates())
        if step_size is None:
            step_size = (x_max - x_min)/num_ticks
        ticks_at = np.arange(start=x_min, stop=x_max, step=step_size)
    time_beta_plot = df.boxplot(
        column=y,
        by=time_col,
        whis=[2.5, 97.5],
        positions=df[time_col].drop_duplicates(),
        ax=ax,
        return_type='dict',
        showcaps=False,
        patch_artist=fill,
    )
    f.suptitle('')
    _ = plt.xticks(rotation="vertical")
    _ = plt.xlabel(xlabel)
    _ = plt.ylabel(ylabel)
    _ = plt.title('')
    _ = ax.xaxis.set_ticks(ticks_at)
    _ = ax.xaxis.set_ticklabels(
         [r"%d" % (int(round(x))) for x in ticks_at])
    
    if ylim:
        _ = plt.ylim(ylim)

    if dict(**kwargs):
        _ = plt.setp(time_beta_plot[y]['boxes'], **kwargs)
        _ = plt.setp(time_beta_plot[y]['medians'], **kwargs)
        _ = plt.setp(time_beta_plot[y]['whiskers'], **kwargs)

def plot_time_betas(models=None, df=None, element='beta_time',
                    y='beta', trans=None, coefs=None, x='timepoint_end_col',
                    by=['model_cohort','coef'], timepoint_id_col=None, timepoint_end_col=None, 
                    subplot=None, ticks_at=None, ylabel=None, xlabel='time',
                    num_ticks=10, step_size=None, fill=True, alpha=0.5, pal=None,
                    value_name='beta', **kwargs):
    if df is None:
        df = extract_time_betas(models=models, element=element, coefs=coefs,
                                     value_name=value_name, timepoint_id_col=timepoint_id_col,
                                     timepoint_end_col=timepoint_end_col)
        timepoint_id_col, timepoint_end_col = _get_timepoint_cols(models=models,
                                                              timepoint_id_col=timepoint_id_col,
                                                              timepoint_end_col=timepoint_end_col)
        logger.debug('timepoint_id_col set to {}'.format(timepoint_id_col))
        logger.debug('timepoint_end_col set to {}'.format(timepoint_end_col))
    if trans:
        trans_name = '{}({})'.format(trans.__name__, y)
        df[trans_name] = trans(df[y])
        y = trans_name
    if by:
        if not pal:
            num_grps = len(df.drop_duplicates(subset=by).loc[:, by].values)
            pal = _get_color_palette(num_grps)
        legend_handles = list()
        i = 0
        if not subplot:
            subplot = plt.subplots(1, 1)
        for grp, grp_df in df.groupby(by):
            _plot_time_betas(df=grp_df.copy(),
                             timepoint_id_col=timepoint_id_col, timepoint_end_col=timepoint_end_col,
                             num_ticks=num_ticks, step_size=step_size, ticks_at=ticks_at,
                             x=x, y=y, color=pal[i], subplot=subplot, alpha=alpha, fill=fill, **kwargs)
            legend_handles.append(mpatches.Patch(color=pal[i], label=grp))
            i = i+1
        plt.legend(handles=legend_handles)
        plt.show()
    else:
        _plot_time_betas(df=df, num_ticks=num_ticks, step_size=step_size, ticks_at=ticks_at,
                         timepoint_id_col=timepoint_id_col, timepoint_end_col=timepoint_end_col,
                         x=x, y=y, subplot=subplot, alpha=alpha, fill=fill, **kwargs)

def _get_sample_ids_single_model(model, sample_col=None, sample_id_col=None):
    if not sample_col:
        sample_col = model['sample_col']
        if not sample_col:
            raise ValueError('sample_col was not given and is also not provided to the model. This is a required input')
    if not sample_id_col:
        sample_id_col = model['sample_id_col']
        if not sample_id_col:
            raise ValueError('sample_id_col was not given and is also not provided to the model. This is a required input')
    patient_sample_ids = model['df'].loc[:,[sample_col, sample_id_col]].drop_duplicates().sort_values(sample_id_col)
    patient_sample_ids['model_cohort'] = model['model_cohort']
    return patient_sample_ids


def get_sample_ids(models, sample_col='patient_id'):
    data = [_get_sample_ids_single_model(model=model, sample_col=sample_col) for model in models]
    return pd.concat(data)


def _prep_pp_data_single_model(model, time_element='y_hat_time', event_element='y_hat_event',
                               sample_col=None, time_col='event_time', event_col='event_status',
                               join_with='df_all', sample_id_col=None):
    patient_sample_ids = _get_sample_ids_single_model(model=model, sample_col=sample_col, sample_id_col=sample_id_col)
    if not sample_col:
        sample_col = model['sample_col']
    pp_event_time = extract_params_long(models=[model],
                                       element=time_element,
                                       varnames=patient_sample_ids[sample_col].values,
                                       )
    pp_event_time.rename(columns={'value': time_col, 'variable': sample_col}, inplace=True)
    pp_event_status = extract_params_long(models=[model],
                                       element=event_element,
                                       varnames=patient_sample_ids[sample_col].values,
                                       )
    pp_event_status.rename(columns={'value': event_col, 'variable': sample_col}, inplace=True)
    pp_data = pd.merge(pp_event_time, pp_event_status, on=['iter', sample_col, 'model_cohort'])
    if join_with:
            pp_data = pd.merge(pp_data, model[join_with], on=sample_col, suffixes=['','_original'])
    return pp_data


def prep_pp_data(models, time_element='y_hat_time', event_element='y_hat_event', event_col='event_status', time_col='event_time', **kwargs):
    data = [_prep_pp_data_single_model(model=model, event_element=event_element, time_element=time_element, event_col=event_col, time_col=time_col, **kwargs)
            for model in models]
    data = pd.concat(data)
    data.sort_values([time_col, 'iter'], inplace=True)
    return data


def prep_pp_survival_data(models, time_element='y_hat_time', event_element='y_hat_event',
                         time_col='event_time', event_col='event_status',
                         by=None,
                         **kwargs):
    pp_data = prep_pp_data(models, time_element=time_element, event_element=event_element, time_col=time_col, event_col=event_col, **kwargs)
    groups = ['iter', 'model_cohort']
    if by:
        groups.append(by)
    pp_surv = pp_data.groupby(groups).apply(
         lambda df: _summarize_survival(df, time_col=time_col, event_col=event_col))
    pp_surv.reset_index(inplace=True)
    return pp_surv

def _plot_pp_survival_data(pp_surv, time_col='event_time', survival_col='survival',
                           num_ticks=10, step_size=None, ticks_at=None, subplot=None,
                           ylabel='Survival %', xlabel='Days', label='posterior predictions',
                           fill=True, **kwargs):
    pp_surv.sort_values(time_col, inplace=True)
    if not subplot:
        f, ax = plt.subplots(1, 1)
    else:
        f, ax = subplot
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
        return_type='dict',
        showcaps=False,
        patch_artist=fill,
    )
    f.suptitle('')
    _ = plt.ylim([0, 1])
    _ = plt.xticks(rotation="vertical")
    _ = plt.xlabel(xlabel)
    _ = plt.ylabel(ylabel)
    _ = plt.title('')
    _ = ax.xaxis.set_ticks(ticks_at)
    _ = ax.xaxis.set_ticklabels(
         [r"%d" % (int(round(x))) for x in ticks_at])

    if dict(**kwargs):
        _ = plt.setp(survival_plot[survival_col]['boxes'], **kwargs)
        _ = plt.setp(survival_plot[survival_col]['medians'], **kwargs)
        _ = plt.setp(survival_plot[survival_col]['whiskers'], **kwargs)


def _get_color_palette(n):
    """ Pick a color palette given number of values
        Returns an array containing color inputs for each value
    """
    if n <= 4:
        color_list = plt.cm.Set1(np.linspace(0, 1, n))
    else:
        color_list = plt.cm.viridis(np.linspace(0, 1, n))
    return color_list

def plot_pp_survival(models, time_element='y_hat_time', event_element='y_hat_event',
                     num_ticks=10, step_size=None, ticks_at=None, time_col='event_time',
                     event_col='event_status', fill=True, by=None, alpha=0.5, pal=None,
                     subplot=None, **kwargs):
    pp_surv = prep_pp_survival_data(models, time_element=time_element,
                                    event_element=event_element, time_col=time_col,
                                    event_col=event_col, by=by)
    if by:
        if not pal:
            num_grps = len(pp_surv.drop_duplicates(subset=by)[by].values)
            pal = _get_color_palette(num_grps)
        legend_handles = list()
        i = 0
        if not subplot:
            subplot = plt.subplots(1, 1)
        for grp, df in pp_surv.groupby(by):
            _plot_pp_survival_data(df.copy(), num_ticks=num_ticks, step_size=step_size, ticks_at=ticks_at,
                                  time_col=time_col, color=pal[i], subplot=subplot, alpha=alpha, fill=fill, **kwargs)
            legend_handles.append(mpatches.Patch(color=pal[i], label=grp))
            i = i+1
        plt.legend(handles=legend_handles)
        plt.show()
    else:
        _plot_pp_survival_data(pp_surv, num_ticks=num_ticks, step_size=step_size,
                               ticks_at=ticks_at, time_col=time_col, alpha=alpha, fill=fill, **kwargs)


def plot_observed_survival(df, event_col, time_col, label='observed', *args, **kwargs):
    actual_surv = _summarize_survival(df=df, time_col=time_col, event_col=event_col)
    plt.plot(actual_surv[time_col], actual_surv['survival'], label=label, *args, **kwargs)

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


def plot_coefs(models, element='coefs', force_direction=None, trans=None, **kwargs):
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
        - 'beta_time'
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
    
    if element=='beta_time':
        return plot_time_betas(models=models, element=element, trans=trans, **kwargs)
    
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


def filter_stan_summary(stan_fit, pars=None, remove_nan=False):
    """ Filter stan fit summary, for the set of parameters in `pars`.
        See ?pystan.summary for details about summary stats given.
    
        Parameters
        -------

        stan_fit: 
            StanFit object for which posterior draws are desired to be summarized
        pars: (list, optional)
            list of strings used to filter parameters. Passed directly to `pystan.summary`.
            default: return all parameters
        remove_nan: (bool, optional)
            whether to remove (and report on) NaN values for Rhat. These are problematic for distplot.

        Returns 
        -------
        pandas dataframe containing summary stats for posterior draws of selected parameters


    """
    if isinstance(stan_fit, list):
        if len(stan_fit)>1:
            logger.warning('More than one model passed to `filter_stan_summary`. Using only the first.')
        stan_fit = stan_fit[0]['fit']
        ## else: assume stan_fit was passed correctly
    if pars:
        fitsum = stan_fit.summary(pars=pars)
    else:
        fitsum = stan_fit.summary()
    df = pd.DataFrame(fitsum['summary'], columns=fitsum['summary_colnames'], index=fitsum['summary_rownames'])
    if remove_nan:
        ## most of NaN values are Rhat
        ## remove & report on their frequency if remove_nan == True
        df_nan_rows = pd.isnull(df).any(1)
        if any(df_nan_rows):
            logger.info('Warning - {} rows removed due to NaN values for Rhat. This may indicate a problem in your model estimation.'.format(df_nan_rows[df_nan_rows].count()))
            df = df[~df_nan_rows]
    return df.loc[:,['mean','se_mean','sd','2.5%','50%','97.5%','Rhat']]


def print_stan_summary(stan_fit, pars=None):
    """ Convenience function to print stan fit summary, for the set of parameters in `pars`.

        Parameters
        -------

        stan_fit: 
            StanFit object for which posterior draws are desired to be summarized
        pars: (optional)
            list of strings used to filter parameters. Passed directly to `pystan.summary`.
            default: return all parameters
    """
    print(filter_stan_summary(stan_fit=stan_fit, pars=pars).to_string())


def plot_stan_summary(stan_fit, pars=None, metric='Rhat'):
    """ Plot distribution of values in stan fit summary, for the set of parameters in `pars`.

        Primary use case is to summarize Rhat estimates for set of parameters, as a quick check of convergence.

        Parameters
        -------

        stan_fit: 
            StanFit object for which posterior draws are desired to be summarized
        pars: (list of str, optional)
            list of strings used to filter parameters. Passed directly to `pystan.summary`.
            default: return all parameters
        metric: (str, optional)
            the name of the metric to plot, as one of: ['mean','se_mean','sd','2.5%','50%','97.5%','Rhat']
            default: `Rhat`
    """
    df = filter_stan_summary(stan_fit=stan_fit, pars=pars, remove_nan=True)
    if not metric in df.columns:
        raise ValueError('Invalid metric ({}). Should be one of {}'.format(metric, '.'.join(df.columns)))
    sb.distplot(df[metric])
