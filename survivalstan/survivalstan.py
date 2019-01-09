
import patsy
import stanity
import pandas as pd
import numpy as np
import logging
# n.b. necessary to import surv directly b/c of how patsy formulas work
# see: http://patsy.readthedocs.io/en/latest/stateful-transforms.html
# which reads
#         Currently the rule is that you must access the stateful transform
#         function using a simple, bare variable reference, without any dots
#         or other lookups
from .formulas import surv  # noqa: F401
from . import formulas

logger = logging.getLogger(__name__)


def fit_stan_survival_model(df=None,
                            formula=None,
                            event_col=None,
                            model_code=None,
                            file=None,
                            model_cohort='survival model',
                            time_col=None,
                            sample_id_col=None,
                            sample_col=None,
                            group_id_col=None,
                            group_col=None,
                            timepoint_id_col=None,
                            timepoint_end_col=None,
                            make_inits=None,
                            stan_data=dict(),
                            grp_coef_type=None,
                            FIT_FUN=stanity.fit,
                            drop_intercept=True,
                            input_data=None,
                            *args, **kwargs):
    """Prepare data & fit a survival model using Stan

    This function wraps a number of steps into one function:

     1. Prepare input data dictionary for Stan
        - calls `SurvivalStanData` with user-provided formulas & df
        - (can be overridden using the `input_data` parameter)
     2. Compiles & optionally caches compiled stan code
     3. Fits model to data
     4. Tries the following functions on the resulting fit object:
      - `stanity.psisloo` to summarize model fit using LOO-PSIS approximation
      - extract posterior draws for beta coefficients
        (if model contains `beta` parameter)
      - extract posterior draws for grouped-beta coefficients (if applicable)

    Parameters:
       df (pandas DataFrame):  The data frame containing input data to Survival
          model.
       formula (chr): Patsy formula to use for covariates.
          E.g 'met_status + pd_l1'
       event_col (chr): name of column containing event status.
          Will be coerced to boolean
       model_code (chr): stan model code to use.
       file (chr): path to stan file (if model_code not given)
       *args, **kwargs: passed to FIT_FUN (stanity.fit or replacement)

       model_cohort (chr): description of this model fit, to be used
          when plotting or summarizing output
       time_col (chr): name of column containing event time -- used
          for parameteric models
       sample_id_col (chr): name of column containing numeric sample ids
          (1-indexed & sequential)
       sample_col (chr): name of column containing sample descriptions - will
          be converted to an ID
       group_id_col (chr): name of column containing numeric group ids
          (1-indexed & sequential)
       group_col (chr): name of column containing group descriptions - will
          be converted to an ID
       timepoint_id_col (chr): name of column containing timepoint
          ids (1-indexed & sequential)
       timepoint_end_col (chr): name of column containing end times for each
          timepoint (will be converted to an ID)
       stan_data (dict): extra params passed to stan data object
       grp_coef_type (chr): type of group coef specified, if using a
          varying-coef model
              Can be one of:
              - 'None' (default): guess group coef orientation from data.
                    Works except in case where M (covariates) == G (groups)
              - 'matrix': grp_beta defined as `matrix[M, G] grp_beta;`
              - 'vector-of-vectors': grp_beta defined
                    as `vector[M] grp_beta[G];`
       drop_intercept (bool): whether to drop the intercept term from
           the model matrix (default: True)

    Returns:
       dictionary of results objects.

       Contents::
          df: Pandas data frame containing input data,
              filtered to non-missing obs & with ID variables created
          x_df: Covariate matrix passed to Stan
          x_names: Column names for the covariate matrix passed to Stan
          data: List passed to Stan - contains dimensions, etc.
          fit: pystan fit object returned from Stan call
          coefs: posterior draws for coefficient values
          loo: psis-loo object returned for fit model. Used for model
              comparison & summary
          model_cohort: description of this model and/or cohort on
              which the model was fit
          df_all: input df given, with calculated values included
          sample_col: name of column (in df_all) used to identify the sample
          sample_id_col: name of column containing numeric id
              derived from the sample
          timepoint_end_col: name of column (in df_all) used to determine
              end-time of 'long' data, if relevant
          timepoint_id_col: name of column containing numeric id derived
              from timepoint_end_col

    Raises:
       AttributeError, KeyError

    Example:

    >>> testfit = fit_stan_survival_model(
                model_file = stanmodels.stan.pem_survival_model,
                formula = '~ met_status + pd_l1',
                df = dflong,
                sample_col = 'patient_id',
                timepoint_end_col = 'end_time',
                event_col = 'end_failure',
                model_cohort = 'PEM survival model',
                iter = 30000,
                chains = 4,
            )
    >>> print(testfit['fit'])
    >>> seaborn.boxplot(x = 'value', y = 'variable', data = testfit['coefs'])

    """

    if model_code is None:
        if file is None:
            raise AttributeError('Either model_code or file is required.')

    if input_data is None:
        input_data = SurvivalStanData(df=df,
                                      formula=formula,
                                      time_col=time_col,
                                      event_col=event_col,
                                      sample_id_col=sample_id_col,
                                      sample_col=sample_col,
                                      group_id_col=group_id_col,
                                      group_col=group_col,
                                      timepoint_id_col=timepoint_id_col,
                                      timepoint_end_col=timepoint_end_col,
                                      drop_intercept=drop_intercept,
                                      **stan_data
                                      )
    x_df = input_data.x_df
    df_nonmiss = input_data.df_nonmiss

    if make_inits:
        kwargs = dict(kwargs, init=make_inits(input_data.data))

    survival_fit = FIT_FUN(
        model_code=model_code,
        file=file,
        data=input_data.data,
        *args,
        **kwargs
        )

    try:
        beta_coefs = pd.DataFrame(
            survival_fit.extract()['beta'],
            columns=x_df.columns
            )
        beta_coefs.reset_index(0, inplace=True)
        beta_coefs = beta_coefs.rename(columns={'index': 'iter'})
        beta_coefs = pd.melt(beta_coefs, id_vars=['iter'])
        beta_coefs['exp(beta)'] = np.exp(beta_coefs['value'])
        beta_coefs['model_cohort'] = model_cohort
    except:
        beta_coefs = None

    # prep by-group coefs if group specified
    if input_data.group_id_col:
        try:
            grp_names = input_data.get_group_names()
            grp_coefs = _extract_grp_coefs(survival_fit=survival_fit,
                                           element='grp_beta',
                                           grp_coef_type=grp_coef_type,
                                           grp_names=grp_names,
                                           columns=x_df.columns,
                                           input_data=input_data.data,
                                           model_cohort=model_cohort
                                           )
        except:
            logger.warning('Error extracting grp_coefs from model fit')
            grp_coefs = None
    else:
        grp_coefs = beta_coefs
        if grp_coefs is not None:
            grp_coefs['group'] = 'Overall'

    try:
        loo = stanity.psisloo(survival_fit.extract()['log_lik'])
    except:
        loo = None

    if not sample_id_col:
        sample_id_col = None
    if not sample_col:
        sample_col = None
    if not timepoint_id_col:
        timepoint_id_col = None
    if not timepoint_end_col:
        timepoint_end_col = None

    return {
        'df': df_nonmiss,
        'x_df': x_df,
        'x_names': x_df.columns,
        'data': input_data.data,
        'fit': survival_fit,
        'coefs': beta_coefs,
        'grp_coefs': grp_coefs,
        'loo': loo,
        'model_cohort': model_cohort,
        'df_all': input_data.df,
        'sample_col': input_data.sample_col,
        'sample_id_col': input_data.sample_id_col,
        'timepoint_id_col': input_data.timepoint_id_col,
        'timepoint_end_col': input_data.timepoint_end_col,
    }


class SurvivalStanData:
    'Input data representing a survival model in survivalstan'

    def __init__(self,
                 df, formula, event_col=None,
                 time_col=None,
                 sample_id_col=None, sample_col=None,
                 group_id_col=None, group_col=None,
                 timepoint_id_col=None, timepoint_end_col=None,
                 drop_intercept=True,
                 **kwargs):
        # capture input params
        self.df = df
        self.formula = formula
        self.event_col = event_col
        self.time_col = time_col
        self.group_col = group_col
        self.timepoint_end_col = timepoint_end_col
        self.sample_col = sample_col
        self.sample_id_col = sample_id_col
        self.group_id_col = group_id_col
        self.timepoint_id_col = timepoint_id_col
        self.drop_intercept = drop_intercept
        # prepare data
        self.prep_survival_formula()
        self.prep_design_info()
        self.prep_df_nonmiss()
        self.prep_stan_data(**kwargs)

    def prep_survival_formula(self):
        ''' Process inputs & convert to new survival-formula syntax
        '''
        # does given formula have a LHS component & use surv syntax?
        if formulas.formula_uses_surv(self.formula, self.df):
            self.surv_formula = self.formula
        elif formulas.formula_has_lhs(self.formula):
            raise ValueError('Given formula has LHS component not using `surv`'
                             ' syntax. Please see ?survivalstan.formulas.surv')
        else:
            rhs_formula = formulas.SurvivalModelDesc(self.formula).rhs
            self.surv_formula = (formulas
                                 .gen_surv_formula(
                                   rhs_formula=rhs_formula,
                                   event_col=self.event_col,
                                   group_col=self.group_col,
                                   sample_col=self.sample_col,
                                   timepoint_end_col=self.timepoint_end_col,
                                   time_col=self.time_col))

    def prep_design_info(self):
        ''' Prepare x and y design matrices
        '''
        (self.y, self.x) = patsy.dmatrices(
            formula_like=formulas.SurvivalModelDesc(self.surv_formula),
            data=self.df,
            )
        surv_factor = self.y.design_info.terms[0].factors[0]
        if surv_factor._is_survival:
            args = surv_factor.code_args
            if 'subject' in args.keys():
                self.sample_col = args['subject']
            if 'group' in args.keys():
                self.group_col = args['group']
            if surv_factor._type == 'long':
                if 'time' in args.keys():
                    self.timepoint_end_col = args['time']

    def prep_df_nonmiss(self):
        ''' Create x_df and df_nonmiss
        '''
        # prep dataframe containing RHS vars (x_df)
        x_df = pd.DataFrame(self.x, columns=self.x.design_info.term_names)
        if len(x_df.columns) > 1 and self.drop_intercept:
            x_df = x_df.ix[:, x_df.columns != 'Intercept']
        self.x_df = x_df
        # prep dataframe containing LHS vars (y_df)
        self.y_df = self.y.design_info.terms[0].factors[0]._meta_data['df']
        # prep df_nonmiss containing both
        if self.y_df.shape[0] != self.x_df.shape[0]:
            raise ValueError('x and y dataframes have different lengths.')
        self.x_df.reset_index(drop=True, inplace=True)
        self.y_df.reset_index(drop=True, inplace=True)
        self.df_nonmiss = pd.concat([self.y_df, self.x_df], axis=1)
        if self.df_nonmiss.dropna().shape != self.df_nonmiss.shape:
            raise ValueError('Missing data in df_nonmiss')
        self._update_df_with_ids()

    def _update_df_with_ids(self):
        ''' add ID decodes to df_nonmiss
        '''
        # name object decode for user-provided subject, group & timepoint cols
        decode_objects = {'subject_id': self.sample_col,
                          'group_id': self.group_col,
                          'timepoint_id': self.timepoint_end_col,
                          }
        mdata = self.y.design_info.terms[0].factors[0]._meta_data
        for decode in decode_objects.keys():
            if decode in mdata.keys():
                decode_df = mdata[decode]
                if decode_df.dropna().shape != decode_df.shape:
                    logger.warning('NA values in decode of {}'.format(decode))
                decode_df.dropna(inplace=True)
                decode_id_col = decode
                decode_id_rename = '_{}'.format(decode_id_col)
                decode_col = decode_objects[decode]
                decode_df.rename(columns={'id': decode_id_rename,
                                          'value': decode_col},
                                 inplace=True)
                self.df_nonmiss.rename(columns={decode_id_col:
                                                decode_id_rename},
                                       inplace=True)
                self.df_nonmiss = pd.merge(self.df_nonmiss, decode_df,
                                           on=decode_id_rename, how='left')
                if decode == 'subject_id':
                    self.sample_id_col = decode_id_rename
                    self.sample_col = decode_col  # not necessary, may be safer
                if decode == 'group_id':
                    self.group_id_col = decode_id_rename
                    self.group_col = decode_col  # also not necessary
                    self._prep_grp_names()
                if decode == 'timepoint_id':
                    self.timepoint_id_col = decode_id_rename
                    self.timepoint_end_col = decode_col  # also not necessary
                    self._prep_timepoint_df()

    def prep_stan_data(self, **kwargs):
        ''' Prep data dictionary to pass to stan.fit
        '''
        # LHS stan_data prepared by SurvivalModelDesc
        self.data = self.y.design_info.terms[0].factors[0]._stan_data
        # RHS stan_data from x_df
        # use x_df for now to handle drop_intercept correctly
        self.data.update({
            'x': self.x_df.values,
            'M': self.x_df.shape[1]
        })
        if dict(**kwargs):
            self.data.update(dict(**kwargs))

    def _prep_grp_names(self):
        ''' Populate grp_names attribute
        '''
        self.grp_names = (self
                          .df_nonmiss
                          .loc[
                           ~(self
                             .df_nonmiss[[self.group_id_col]]
                             .duplicated())]
                          .sort_values(self.group_id_col)[self.group_col]
                          .values)

    def get_group_names(self):
        if not self.group_id_col:
            return(None)
        return(list(self.grp_names))

    def _prep_timepoint_df(self):
        ''' Add timepoint-id-related data to self.timepoint_df
        '''
        unique_timepoints = _prep_timepoint_dataframe(
                             self.df_nonmiss,
                             timepoint_id_col=self.timepoint_id_col,
                             timepoint_end_col=self.timepoint_end_col
                             )
        unique_timepoints.reset_index(inplace=True)
        self.timepoint_df = unique_timepoints


def _extract_grp_coefs(survival_fit, element, grp_coef_type, grp_names,
                       columns, input_data, model_cohort):
    """ Helper function to extract grp coefs summary data
    """
    grp_coefs_extract = survival_fit.extract()[element]
    # try to guess shape of group-betas
    if not(grp_coef_type):
        grp_coef_type = _guess_grp_coef_type(extract=grp_coefs_extract,
                                             input_data=input_data)
    # process group_coefs according to type
    if grp_coef_type == 'matrix':
        try:
            grp_coefs_data = _format_grp_coefs_matrix(
                              extract=grp_coefs_extract,
                              columns=columns,
                              grp_names=grp_names
                              )
        except:
            raise Exception('unable to format grp coefs as matrix')
    elif grp_coef_type == 'vector-of-vectors':
        try:
            grp_coefs_data = _format_grp_coefs_vectors(
                               extract=grp_coefs_extract,
                               columns=columns,
                               grp_names=grp_names
                               )
        except:
            raise Exception('unable to format grp coefs as vector-of-vectors')
    elif grp_coef_type == 'unknown':
        print("warning: unable to determine group-coef orientation."
              " Try using arg `grp_coef_type`")
        return(None)
    else:
        print("Invalid `grp_coef_type` -- must be one of 'vector-of-vectors'"
              " or 'matrix'")
        print("Skipping grp coef extraction for now.")
        return(None)

    # process/format grp_coefs data
    grp_coefs = pd.melt(grp_coefs_data, id_vars=['group', 'iter'])
    grp_coefs['exp(beta)'] = np.exp(grp_coefs['value'])
    grp_coefs['group'] = grp_coefs.group.astype('category')
    grp_coefs['model_cohort'] = model_cohort
    return(grp_coefs)


def _format_grp_coefs_matrix(extract, columns, grp_names):
    """ Helper function for format grp_coefs data if in `matrix[M, G]` form
    """
    grp_coefs_data = list()
    i = 0
    for grp in grp_names:
        grp_data = pd.DataFrame(extract[:, :, i], columns=columns)
        grp_data.reset_index(inplace=True)
        grp_data.rename(columns={'index': 'iter'}, inplace=True)
        grp_data['group'] = grp
        grp_coefs_data.append(grp_data)
        i = i+1
    return(pd.concat(grp_coefs_data))


def _format_grp_coefs_vectors(extract, columns, grp_names):
    """ Helper function for format grp_coefs data
        if in `vector[M] grp_beta[G]` form
    """
    grp_coefs_data = list()
    i = 0
    for grp in grp_names:
        grp_data = pd.DataFrame(extract[:, i, :], columns=columns)
        grp_data.reset_index(inplace=True)
        grp_data.rename(columns={'index': 'iter'}, inplace=True)
        grp_data['group'] = grp
        grp_coefs_data.append(grp_data)
        i = i+1
    return(pd.concat(grp_coefs_data))


def _guess_grp_coef_type(extract, input_data):
    """ helper function to determine grp_coefs type from shape
        of returned object
    """
    if input_data['M'] == input_data['G']:
        # unable to determine shape if M == G
        grp_coef_type = 'unknown'
    elif extract.shape[1] == input_data['G']:
        grp_coef_type = 'vector-of-vectors'
    elif extract.shape[2] == input_data['G']:
        grp_coef_type = 'matrix'
    return grp_coef_type


def _prep_timepoint_dataframe(df,
                              timepoint_end_col,
                              timepoint_id_col=None
                              ):
    """ Helper function to take a set of timepoints
        in observation-level dataframe & return
        formatted timepoint_id, end_time, duration

        Returns
        ---------
        pandas dataframe with one record per timepoint_id
            where timepoint_id is the index
            sorted on the index, increasing

    """
    time_df = df.copy()
    time_df.sort_values(timepoint_end_col, inplace=True)
    if not(timepoint_id_col):
        timepoint_id_col = 'timepoint_id'
        time_df[timepoint_id_col] = (time_df[timepoint_end_col]
                                     .astype('category')
                                     .cat.codes + 1)
    time_df.dropna(how='any',
                   subset=[timepoint_id_col, timepoint_end_col],
                   inplace=True)
    time_df = (time_df
               .loc[:, [timepoint_id_col, timepoint_end_col]]
               .drop_duplicates())
    time_df[timepoint_end_col] = time_df[timepoint_end_col].astype(np.float32)
    time_df.set_index(timepoint_id_col, inplace=True, drop=True)
    time_df.sort_index(inplace=True)
    t_durs = time_df.diff(periods=1)
    t_durs.rename(columns={timepoint_end_col: 't_dur'}, inplace=True)
    time_df = time_df.join(t_durs)
    if len(time_df.index) > 1:
        time_df.fillna(inplace=True, value=time_df.loc[1, timepoint_end_col])
    return(time_df)


def extract_grp_baseline_hazard(results, timepoint_id_col='timepoint_id',
                                timepoint_end_col='end_time'):
    """ If model results contain a grp_baseline object, extract & summarize it
    """
    # TODO check if results are by-group
    # TODO check if baseline hazard is computable
    grp_baseline_extract = results['fit'].extract()['grp_baseline']
    coef_group_names = results['grp_coefs']['group'].unique()
    i = 0
    grp_baseline_data = list()
    for grp in coef_group_names:
        grp_base = pd.DataFrame(grp_baseline_extract[:, :, i])
        grp_base_coefs = pd.melt(grp_base,
                                 var_name=timepoint_id_col,
                                 value_name='baseline_hazard')
        grp_base_coefs['group'] = grp
        grp_baseline_data.append(grp_base_coefs)
        i = i+1
    grp_baseline_coefs = pd.concat(grp_baseline_data)
    end_times = _extract_timepoint_end_times(
                 results,
                 timepoint_id_col=timepoint_id_col,
                 timepoint_end_col=timepoint_end_col)
    bs_data = pd.merge(grp_baseline_coefs, end_times, on=timepoint_id_col)
    return(bs_data)


def _extract_timepoint_end_times(results, timepoint_end_col='end_time',
                                 timepoint_id_col='timepoint_id'):
    df_nonmiss = results['df']
    end_times = (df_nonmiss
                 .loc[~df_nonmiss[[timepoint_id_col]].duplicated()]
                 .sort_values(timepoint_id_col)[
                  [timepoint_end_col, timepoint_id_col]])
    return(end_times)


def extract_baseline_hazard(results, element='baseline', timepoint_id_col=None,
                            timepoint_end_col=None):
    """ If model results contain a baseline object, extract & summarize it
    """
    if timepoint_id_col is None:
        timepoint_id_col = results['timepoint_id_col']
    if timepoint_end_col is None:
        timepoint_end_col = results['timepoint_end_col']
    # TODO check if baseline hazard is computable
    baseline_extract = results['fit'].extract()[element]
    baseline_coefs = pd.DataFrame(baseline_extract)
    bs_coefs = pd.melt(baseline_coefs,
                       var_name=timepoint_id_col,
                       value_name='baseline_hazard')
    end_times = _extract_timepoint_end_times(
                 results,
                 timepoint_id_col=timepoint_id_col,
                 timepoint_end_col=timepoint_end_col)
    bs_data = pd.merge(bs_coefs, end_times, on=timepoint_id_col)
    bs_data['model_cohort'] = results['model_cohort']
    return(bs_data)


def prep_data_long_surv(df, time_col, event_col, sample_col=None,
                        event_name=None):
    ''' Convert wide survival dataframe (df) to long format,
        in preparation for modeling using PEM models.

        Returns a pandas DataFrame with original records duplicated for
            each unique failure time observed.
            Each record will have two new columns: 'end_failure' and
              'end_time', indicating the event status (`end_failure`)
              for each unique timepoint (`end_time`).

        Parameters:
            df (pandas.DataFrame):
               Input data containing survival time & status for each subject
            time_col (str):
               name of column containing time to censor/event
            event_col (str or list of strings):
              name of column containing status (1 or True: event, 0 or
               False: censor). If a list is provided, these will be processed
               as multiple event types.
            sample_col (str):
              (optional) column containing sample or subject identifier.
              If given, result will be de-duped so that multiple events within
              a sample are handled correctly.
            event_name (str):
              (optional) column containing description of event type, if
              more than one type of event is observed. If given, multiple
              events per subject will be processed.

        Returns:
            pandas.DataFrame with original records duplicated for each unique
            failure time observed.

                Each record will _include all original covariate values_, plus
                two new columns: 'end_failure' and 'end_time', indicating the
                timepoint-specific event status for each record.

                If multiple events are given (either via a list of event_cols
                or by providing an event_name, the result will contain multiple
                end_failure columns, one for each event type.

    '''
    # process multiple event_names, if given:
    if event_name:
        if not sample_col:
            raise ValueError('Sample col is required to process'
                             ' multiple events')
        df_events = pd.pivot_table(df,
                                   index=[sample_col, time_col],
                                   columns=[event_name],
                                   values=[event_col],
                                   fill_value=False).copy()
        df_events.reset_index(col_level=1, inplace=True)
        df_events.columns = df_events.columns.droplevel(0)
        event_cols = list(df[event_name].unique())
        df_covars = df.loc[:,
                           [column for column in df.columns
                            if column not in [event_name, event_col]
                            ]
                           ].drop_duplicates().copy()
        assert(all(df_covars.duplicated(subset=[sample_col, time_col])
                   == False))
        df_multi = pd.merge(df_events, df_covars, on=[sample_col, time_col],
                            how='outer')
    else:
        df_multi = df
        event_cols = event_col
    if isinstance(event_cols, list):
        logger.debug('Event col is given as a list;'
                     ' processing multi-event data')
        # start with covariates per subject_id
        df_covars = df_multi.loc[:,
                                 [column for column in df_multi.columns
                                  if column not in event_cols
                                  and column not in time_col]].copy()
        df_covars.drop_duplicates(inplace=True)
        assert(all(df_covars.duplicated(subset=[sample_col]) == False))
        # merge in event-data for each event type
        ldf = None
        for event in event_cols:
            longdata = prep_data_long_surv(df_multi,
                                           event_col=event,
                                           time_col=time_col,
                                           sample_col=sample_col
                                           )
            longdata = (longdata
                        .loc[:, [sample_col, 'end_time', 'end_failure']]
                        .copy())
            longdata.rename(columns={'end_failure': 'end_{}'.format(event)},
                            inplace=True)
            if ldf is None:
                ldf = longdata
            else:
                ldf = pd.merge(ldf,
                               longdata,
                               on=[sample_col, 'end_time'],
                               how='outer')
        with_covars = pd.merge(ldf, df_covars, on=sample_col, how='outer')
        return with_covars
    # identify distinct failure/censor times
    failure_times = df[time_col].unique()
    ftimes = pd.DataFrame({'end_time': failure_times, 'key': 1})
    # cross join failure times with each observation
    df['key'] = 1
    dflong = pd.merge(df, ftimes, on='key')
    del dflong['key']
    # identify end-time & end-status for each sample*failure time

    def gen_end_failure(row):
        if row[time_col] > row['end_time']:
            # event not yet occurred (time_col is after this timepoint)
            return False
        if row[time_col] == row['end_time']:
            # event during (==) this timepoint
            return row[event_col]
        if row[time_col] < row['end_time']:
            # event already occurred (time_col is before this timepoint)
            return np.nan
    dflong['end_failure'] = dflong.apply(
                             lambda row: gen_end_failure(row),
                             axis=1)
    # confirm total number of non-censor events hasn't changed
    if not(sum(dflong.end_failure.dropna()) == sum(df[event_col].dropna())):
        print('Warning: total number of events has changed from {0} to {1}'
              .format(sum(df[event_col]), sum(dflong.end_failure)))
    # remove timepoints after failure/censor event
    dflong = dflong.query('end_time <= {0}'.format(time_col)).copy()
    # if sample_col is given, remove duplicates
    # induced in case of multiple events
    if sample_col:
        dflong['_rank'] = (dflong
                           .groupby([sample_col, 'end_time'])[time_col]
                           .rank())
        dflong = dflong.query('_rank == 1')
        del dflong['_rank']
    return(dflong)


def make_weibull_survival_model_inits(stan_input_dict):
    def f():
        m = {
            'tau_s_raw': abs(np.random.normal(0, 1)),
            'tau_raw': abs(np.random.normal(0, 1, stan_input_dict['M'])),
            'alpha_raw': np.random.normal(0, 0.1),
            'beta_raw': np.random.normal(0, 1, stan_input_dict['M']),
            'mu': np.random.normal(0, 1),
        }
        return m
    return f
