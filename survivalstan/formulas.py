import pandas as pd
import patsy
import sys
import numpy as np

from .survivalstan import _prep_timepoint_dataframe

import logging
logger = logging.getLogger(__name__)

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
        return pd.DataFrame({'id': self.ids, 'value': self.values})

as_id = patsy.stateful_transform(Id)

class SurvData(pd.DataFrame):
    ''' patsy.DesignMatrix representing survival data output '''
    survival_type = 'wide'

    def __init__(self, *args, stan_data=dict(), meta_data=dict(), **kwargs):
        super().__init__(*args, **kwargs)
        self.stan_data = stan_data
        self.meta_data = meta_data

class LongSurvData(SurvData):
    ''' pd.DataFrame representing survival data with endpoint_time_id, event_status & subject_id '''
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
            raise ValueError('Invalid parameter: {}'.format(','.join(bad_keys)))
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

    def _prep_long(self, timepoint_id, event_status, subject_id, group_id=None, **kwargs):
        if patsy.util.have_pandas:
            dm = {'timepoint_id': timepoint_id,
                  'event_status': event_status,
                  'subject_id': subject_id
                  }
            if group_id is not None:
                dm.update({'group_id': group_id})
            dm = pd.DataFrame(dm)
            dm.index = event_status.index
        else:
            if group_id is not None:
                dm = np.append(timepoint_id, event_status, subject_id, group_id, 1)
            else:
                dm = np.append(timepoint_id, event_status, subject_id, 1)
        return LongSurvData(dm, **kwargs)

    def _prep_wide(self, time, event_status, group_id=None, **kwargs):
        if patsy.util.have_pandas:
            dm = {'time': time,
                 'event_status': event_status,
                 }
            if group_id is not None:
                dm.update({'group_id': group_id})
            dm = pd.DataFrame(dm)
            dm.index = time.index
        else:
            if group_id is not None:
                dm = np.append(time, event_status, group_id, 1)
            else:
                dm = np.append(time, event_status, 1)
        return SurvData(dm, **kwargs)

    def transform(self, time, event_status, **kwargs):
        kwargs = self._check_kwargs(**kwargs)
        meta_data = dict()
        stan_data = dict()
        if 'subject' in kwargs.keys():
            subject_id = self.subject_id.transform(kwargs['subject'])
            timepoint_id = self.timepoint_id.transform(time)
            meta_data.update({'timepoint_id': self.timepoint_id.decode_df(),
                              'subject_id': self.subject_id.decode_df()})
            stan_data.update(self._prep_timepoint_standata(self.timepoint_id.decode_df()))
            stan_data.update({'S': self.subject_id.len()})
        if 'group' in kwargs.keys():
            group_id = self.group_id.transform(kwargs['group'])
            stan_data.update({'G': len(self.group_id.len())})
            meta_data.update({'group_id': self.group_id.decode_df()})
        else:
            group_id = None

        if self._type == 'long':
            return(self._prep_long(timepoint_id=timepoint_id, event_status=event_status,
                                  subject_id=subject_id, group_id=group_id,
                                  meta_data=meta_data, stan_data=stan_data)
                  )
        elif self._type == 'wide':
            return(self._prep_wide(time=time, event_status=event_status, group_id=group_id,
                                  meta_data=meta_data, stan_data=stan_data))

surv = patsy.stateful_transform(Surv)

class SurvivalFactor(patsy.EvalFactor):
    ''' A factor object to encode LHS variables
        for Survival Models, including model type
    '''
    _is_survival = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class = None

    def eval(self, *args, **kwargs):
        result = super().eval(*args, **kwargs)
        try:
            self._class = result.__class__
        except:
            logger.warning('Outcome class could not be determined')
        if isinstance(result, SurvData):
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
        self.lhs, self.rhs = re.split(string=formula, pattern='~', maxsplit=1)
        self.lhs_termlist = [patsy.Term([SurvivalFactor(self.lhs)])]
        self.rhs_termlist = patsy.ModelDesc.from_formula(self.rhs).rhs_termlist

    def __patsy_get_model_desc__(self, eval_env):
        return patsy.ModelDesc(self.lhs_termlist, self.rhs_termlist)


