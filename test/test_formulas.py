from survivalstan.formulas import *
import survivalstan
from nose.tools import ok_, eq_
from numpy import array_equal

def get_test_data():
    ''' Return test data for patsy formula testing
    '''
    data = survivalstan.sim.sim_data_jointmodel(N=100)
    df = pd.merge(data['events'].query('event_name == "death"'),
                  data['covars'], on='subject_id')
    return(df)

def test_as_id_str():
    ''' Test that as_id uniquely enumerates strings
    '''
    res = as_id(np.array(['a','b','a','c']))
    ok_(array_equal(res, [1, 2, 1, 3]))

def test_as_id_str_alpha():
    ''' Test that as_id uniquely enumerates strings in sort order
    '''
    res = as_id(np.array(['b','a','b','c']))
    ok_(array_equal(res, [2, 1, 2, 3]))

def test_as_id_int():
    ''' Test that as_id uniquely enumerates integers
    '''
    res = as_id(np.array([10, 2, 10, 8]))
    ok_(array_equal(res, [3, 1, 3, 2]))

def test_as_id_formula():
    ''' Test that as_id enumerates strings within a patsy formula
    '''
    test_formula = 'event_value + as_id(time) + as_id(subject_id) ~ X1 + X2'
    df = get_test_data()
    y, X = patsy.dmatrices(formula_like=test_formula, data=df)
    res = pd.DataFrame(y)
    # should have 3 columns & same number rows as df
    eq_(res.shape[1], 3)
    eq_(res.shape[0], df.shape[0])
    # check valid ids
    check_valid_id(res[1], ref=df['time'])
    check_valid_id(res[2], ref=df['subject_id'])

def is_sequential(x):
    it = (int(el) for el in sorted(set(x)))
    first = next(it)
    return all(a == b for a, b in enumerate(it, first + 1))

def test_is_sequential():
    x = [1, 2, 3, 3, 5, 4]
    ok_(is_sequential(x))
    y = [1, 3, 4]
    ok_(not(is_sequential(y)))

def check_valid_id(x, ref=None):
    ''' helper function to validate whether x is an ID
    '''
    ok_(is_sequential(x))
    eq_(np.min(x), 1)
    # TODO test one-to-one & onto relationship
    if ref is not None:
        eq_(np.max(x), len(x.unique()))
    return(True)

def test_surv_df():
    ''' test that surv stateful transform accepts time & event values
    '''
    df = get_test_data()
    res = surv(time=df['time'], event_status=df['event_value'])
    eq_(res.shape[1], 2)
    eq_(res.shape[0], len(df.index))
    ok_(array_equal(res.columns, ['event_status', 'time']))
    eq_(np.sum(res['event_status']), np.sum(df['event_value']))

def test_surv_df_subject():
    ''' test that surv stateful transform includes subject id when given
    '''
    df = get_test_data()
    res = surv(time=df['time'], event_status=df['event_value'],
               subject=df['subject_id'])
    eq_(res.shape[1], 3)
    eq_(res.shape[0], len(df.index))
    ok_(array_equal(res.columns, ['event_status', 'subject_id',
                                  'timepoint_id']))
    check_valid_id(res['subject_id'], ref=df['subject_id'])
    check_valid_id(res['timepoint_id'], ref=df['time'])
    eq_(np.sum(res['event_status']), np.sum(df['event_value']))

def test_surv_df_formula():
    df = get_test_data()
    y, X = patsy.dmatrices('surv(time=time, event_status=event_value) ~ X1',
                           data=df)
    res = pd.DataFrame(y)
    eq_(res.shape[1], 2)
    eq_(res.shape[0], len(df.index))
    eq_(np.sum(res[0]), np.sum(df['event_value']))
    eq_(np.sum(res[1]), np.sum(df['time']))

def test_surv_df_subject_formula():
    df = get_test_data()
    formula = 'surv(time=time, event_status=event_value, subject=subject_id) ~ X1'
    y, X = patsy.dmatrices(formula, data=df)
    res = pd.DataFrame(y)
    # test quality of res
    eq_(res.shape[1], 3)
    eq_(res.shape[0], len(df.index))
    check_valid_id(res[1], ref=df['subject_id'])
    check_valid_id(res[2], ref=df['time'])
    eq_(np.sum(res[0]), np.sum(df['event_value']))
    # test whether class ids are retained when predicting new data
    (y.new, X.new) = patsy.build_design_matrices([y.design_info,
                                                   X.design_info], df.tail()) 
    res2 = pd.DataFrame(y.new)
    resm = pd.merge(res, res2, on=[1,2], how='inner')
    ok_(array_equal(resm['0_x'], resm['0_y']))

