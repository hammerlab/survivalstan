import warnings
import pandas as pd
import patsy
import numpy as np
import re
import logging

warnings.simplefilter(action='ignore', category=UserWarning)
logger = logging.getLogger(__name__)


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
        time_df[timepoint_id_col] = ((time_df[timepoint_end_col]
                                     .astype('category')
                                     .cat
                                     .codes)
                                     + 1)
    time_df.dropna(how='any', subset=[timepoint_id_col, timepoint_end_col],
                   inplace=True)
    time_df = (time_df.loc[:, [timepoint_id_col, timepoint_end_col]]
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


class Id(object):
    def __init__(self, desc='id'):
        self.values = []
        self.desc = desc

    def memorize_chunk(self, x):
        self.values.extend(np.unique(x))

    def memorize_finish(self):
        self.ids = np.arange(len(self.values))+1
        self.lookup = dict(zip(self.values, self.ids))

    def transform(self, x):
        if patsy.util.have_pandas and isinstance(x, pd.Series):
            d = pd.Series([self.lookup[val] for val in x]).astype(int)
            d.index = x.index
            return(d)
        else:
            return np.array([self.lookup[val] for val in x])

    def len(self):
        return len(self.ids)

    def decode_df(self):
        df = pd.DataFrame({'id': self.ids, 'value': self.values})
        df.dropna(inplace=True)
        return df


as_id = patsy.stateful_transform(Id)


class SurvData(pd.DataFrame):
    ''' patsy.DesignMatrix representing survival data output '''
    survival_type = 'UNK'

    def __init__(self, *args, **kwargs):
        if 'stan_data' in kwargs.keys():
            stan_data = kwargs['stan_data']
            del kwargs['stan_data']
        else:
            stan_data = dict()
        if 'meta_data' in kwargs.keys():
            meta_data = kwargs['meta_data']
            del kwargs['meta_data']
        else:
            meta_data = dict()
        super(SurvData, self).__init__(*args, **kwargs)
        self.stan_data = stan_data
        self.meta_data = meta_data


class WideSurvData(SurvData):
    ''' pd.DataFrame representing survival data with one record per subject '''
    survival_type = 'wide'

    def __init__(self, *args, **kwargs):
        super(WideSurvData, self).__init__(*args, **kwargs)
        self._validate_wide_data()

    def _validate_wide_data(self):
        # TODO confirm wide format
        # validate contents of stan_data
        return True


class LongSurvData(SurvData):
    ''' pd.DataFrame representing survival data
    with endpoint_time_id, event_status & subject_id '''
    survival_type = 'long'


class NotValidId(ValueError):
    ''' Class of errors pertaining to invalid Id variables '''


class Surv(object):
    ''' Class representing stateful Survival-type data
    '''
    def __init__(self):
        self.subject_id = Id('subject')
        self.timepoint_id = Id('timepoint')
        self.group_id = Id('group')
        self._type = None
        self._grouped = None
        pass

    def _check_kwargs(self, **kwargs):
        kwargs = dict(**kwargs)
        allowed_kwargs = ['subject', 'group']
        bad_keys = [key not in allowed_kwargs for key in kwargs.keys()]
        if any(bad_keys):
            raise ValueError('Invalid parameter: {}'
                             .format(','.join(bad_keys)))
        return kwargs

    def memorize_chunk(self, time, event_status, **kwargs):
        kwargs = self._check_kwargs(**kwargs)
        if 'subject' in kwargs.keys():
            self._type = 'long'
            self.subject_id.memorize_chunk(kwargs['subject'])
            self.timepoint_id.memorize_chunk(time)
        else:
            self._type = 'wide'
        if 'group' in kwargs.keys():
            self._grouped = True
            self.group_id.memorize_chunk(kwargs['group'])
        else:
            self._grouped = False

    def memorize_finish(self):
        self.subject_id.memorize_finish()
        self.group_id.memorize_finish()
        self.timepoint_id.memorize_finish()

    def _prep_timepoint_standata(self, timepoint_df):
        unique_timepoints = _prep_timepoint_dataframe(
            timepoint_df,
            timepoint_id_col='id',
            timepoint_end_col='value')
        timepoint_input_data = {
            't_dur': unique_timepoints['t_dur'],
            't_obs': unique_timepoints['value'],
            'T': len(unique_timepoints.index)
        }
        return timepoint_input_data

    def _prep_long(self, timepoint_id, event_status, subject_id, group_id=None,
                   stan_data=dict(), meta_data=dict(), **kwargs):
        if not patsy.util.have_pandas:
            raise ValueError('non-pandas use case not supported yet. Please ',
                             'import pandas to use `surv`')
        dm = {'timepoint_id': timepoint_id,
              'event_status': event_status.astype(int),
              'subject_id': subject_id
              }
        if group_id is not None:
            dm.update({'group_id': group_id})
        dm = pd.DataFrame(dm)
        # make sure data.frame columns always ordered alphabetically
        # so:
        # 0. event_status
        # 1. subject_id
        # 2. timepoint_id
        dm.sort_index(axis=1, inplace=True)
        # prep stan_data inputs
        stan_data.update({
                'event': dm['event_status'].values.astype(int),
                't': dm['timepoint_id'].values.astype(int),
                's': dm['subject_id'].values.astype(int),
                'N': len(dm.index),
                })
        if group_id is not None:
            stan_data.update({'g': dm['group_id'].values.astype(int)})
        meta_data.update({'df': dm})
        return LongSurvData(dm,
                            stan_data=stan_data,
                            meta_data=meta_data, **kwargs)

    def _prep_wide(self, time, event_status, group_id=None,
                   stan_data=dict(), meta_data=dict(), **kwargs):
        if not patsy.util.have_pandas:
            raise ValueError('non-pandas usage not yet supported. Please',
                             ' import pandas library to use `surv` syntax')

        # prep pandas dataframe
        dm = {'event_status': event_status.astype(int),
              'time': time,
              }
        if group_id is not None:
            dm.update({'group_id': group_id})
        dm = pd.DataFrame(dm)
        # prep stan_data object
        stan_data.update({'y': dm['time'].values.astype(float),
                          'event': dm['event_status'].values.astype(int),
                          'N': len(dm.index)})
        if group_id is not None:
            stan_data.update({'g': dm['group_id'].values.astype(int)})
        meta_data.update({'df': dm})
        return WideSurvData(dm,
                            stan_data=stan_data,
                            meta_data=meta_data,
                            **kwargs)

    def transform(self, time, event_status, **kwargs):
        kwargs = self._check_kwargs(**kwargs)
        meta_data = dict()
        stan_data = dict()
        if 'subject' in kwargs.keys():
            subject_id = self.subject_id.transform(kwargs['subject'])
            timepoint_id = self.timepoint_id.transform(time)
            meta_data.update({'timepoint_id': self.timepoint_id.decode_df(),
                              'subject_id': self.subject_id.decode_df()})
            stan_data.update(self._prep_timepoint_standata(self
                                                           .timepoint_id
                                                           .decode_df()))
            stan_data.update({'S': self.subject_id.len()})
        if 'group' in kwargs.keys():
            group_id = self.group_id.transform(kwargs['group'])
            stan_data.update({'G': self.group_id.len()})
            meta_data.update({'group_id': self.group_id.decode_df()})
        else:
            group_id = None

        if self._type == 'long':
            return(self._prep_long(timepoint_id=timepoint_id,
                                   event_status=event_status,
                                   subject_id=subject_id,
                                   group_id=group_id,
                                   meta_data=meta_data,
                                   stan_data=stan_data)
                   )
        elif self._type == 'wide':
            return(self._prep_wide(time=time,
                                   event_status=event_status,
                                   group_id=group_id,
                                   meta_data=meta_data,
                                   stan_data=stan_data))


surv = patsy.stateful_transform(Surv)


def _get_args(s):
    ''' Given a string of named code, return dict of named arguments

    Parameters:
        s (string): string in format of:
                    function_name(arg1=val1, arg2=val2, ...)

    Returns:
        if string in format above
           dict containing named parameter values:
                 {'arg1': 'val1', 'arg2': 'val2'}
        note: function_name & unnamed args are ignored
    '''
    pattern = r'(\w[\w\d_]*)\((.*)\)$'
    match = re.match(pattern, s)
    if match and len(match.groups()) == 2:
        d = dict(re.findall(r'(\S+)\s*=\s*(".*?"|[^ ,]+)', match.groups()[1]))
        for name, value in d.items():
            try:
                d[name] = int(value)
            except:
                d[name] = value.strip('\"').strip('\'')
        return(d)
    else:
        raise ValueError('function string {} could not be parsed'.format(s))


class SurvivalFactor(patsy.EvalFactor):
    ''' A factor object to encode LHS variables
        for Survival Models, including model type
    '''
    def __init__(self, *args, **kwargs):
        super(SurvivalFactor, self).__init__(*args, **kwargs)
        self._is_survival = False
        self._class = None

    def eval(self, *args, **kwargs):
        result = super(SurvivalFactor, self).eval(*args, **kwargs)
        try:
            self._class = result.__class__
        except:
            logger.warning('Outcome class could not be determined')
        if isinstance(result, SurvData):
            self.code_args = _get_args(self.code)
            self._is_survival = True
            self._type = result.survival_type
            self._meta_data = result.meta_data
            self._stan_data = result.stan_data

        return result


class SurvivalModelDesc(object):
    ''' A ModelDesc class to force use of SurvivalFactor when encoding LHS
        variables for a SurvivalModel

        Example:

            # simple survival model
            my_formula = SurvivalModelDesc('surv(time=time,
                event_status=event_value) ~ X1')
            y, X = patsy.dmatrices(my_formula, data=df)

            # with a subject id
            my_formula2 = SurvivalModelDesc('surv(time=time,
                    event_status=event_value, subject=subject_id) ~ X1')
            y2, X2 = patsy.dmatrices(my_formula2, data=df)

            # saves information about class & type of survival model
            y2.design_info.terms[0].factors[0]._class
            y2.design_info.terms[0].factors[0]._type
    '''
    def __init__(self, formula):
        self.formula = formula
        try:
            self.lhs, self.rhs = re.split(string=formula,
                                          pattern='~',
                                          maxsplit=1)
        except ValueError:
            self.rhs = formula
            self.lhs = ''
        self.lhs_termlist = [patsy.Term([SurvivalFactor(self.lhs)])]
        self.rhs_termlist = patsy.ModelDesc.from_formula(self.rhs).rhs_termlist

    def __patsy_get_model_desc__(self, eval_env):
        return patsy.ModelDesc(self.lhs_termlist, self.rhs_termlist)


def formula_has_lhs(formula):
    '''return True if formula has LHS. False otherwise
    '''
    surv_model = SurvivalModelDesc(formula)
    return surv_model.lhs.strip() != ''


def formula_uses_surv(formula, df):
    ''' return True if formula uses `surv` syntax & can be successfully parsed
    '''
    if formula_has_lhs(formula):
        surv_model = SurvivalModelDesc(formula)
        (y, X) = patsy.dmatrices(surv_model, df)
        return surv_model.lhs_termlist[0].factors[0]._is_survival
    else:
        return False


def gen_lhs_formula(event_col, time_col=None, group_col=None,
                    sample_col=None, timepoint_end_col=None):
    ''' Construct LHS of formula_like str using `surv` syntax

        Parameters:
            event_col (str):
                name of column containing event status
                  (0:censor/1:observed) or (F/T)
            time_col (str):
                name of column containing time to event
            group_col (str):
                (optional) name of column containing group identifiers,
                   if applicable
            sample_col (str):
                (optional) name of column containing sample or subject
                  identifiers, if applicable
            timepoint_end_col (str):
                (optional) name of column containing timepoint end value,
                   if applicable

        Returns:
            lhs_formula_like (str)
              in surv(param=value, param2=value2, ...) syntax

        Comments:
            Is used by SurvivalStanData class to provide backwards
              compatibility with surv syntax
    '''
    pars = {'event_status': event_col}
    if time_col:
        pars.update({'time': time_col})
    # group column
    if group_col:
        pars.update({'group': group_col})
    # subject column
    if sample_col:
        pars.update({'subject': sample_col})
    # timepoint end col
    if timepoint_end_col:
        pars.update({'time': timepoint_end_col})
    lhs_formula = 'surv({})'.format(','.join(['{}={}'.format(name, value)
                                    for name, value in pars.items()]))
    return(lhs_formula)


def gen_surv_formula(rhs_formula, event_col, time_col=None,
                     group_col=None, sample_col=None, timepoint_end_col=None):
    ''' Construct formula_like str using `surv` syntax

        Parameters:
            rhs_formula (str):
                formula_like (str) for RHS of model spec
            event_col (str):
                name of column containing
                  event status (0:censor/1:observed) or (F/T)
            time_col (str):
                name of column containing time to event
            group_col (str):
                (optional) name of column containing group identifiers,
                  if applicable
            sample_col (str):
                (optional) name of column containing sample or subject
                  identifiers, if applicable
            timepoint_end_col (str):
                (optional) name of column containing timepoint end value,
                  if applicable

        Returns:
            formula_like (str) in
              `surv(param=value, param2=value2, ...) ~ .` syntax

        Comments:
            Is used by SurvivalStanData class
              to provide backwards compatibility with surv syntax
    '''
    lhs_formula = gen_lhs_formula(event_col=event_col, time_col=time_col,
                                  group_col=group_col, sample_col=sample_col,
                                  timepoint_end_col=timepoint_end_col)
    return('~'.join([lhs_formula.strip(), rhs_formula.strip()]))
