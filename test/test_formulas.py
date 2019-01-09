import survivalstan
from survivalstan.formulas import *
from nose.tools import ok_, eq_
from numpy import array_equal

def get_test_data():
    ''' Return test data for patsy formula testing
    '''
    data = survivalstan.sim.sim_data_jointmodel(N=100)
    df = pd.merge(data['events'].query('event_name == "death"'),
                  data['covars'], on='subject_id')
    return(df)

def get_alt_test_data():
    ''' Return test data with event coded as boolean
    '''
    data = get_test_data()
    data['event_value'] = data['event_value'] == 1
    return(data)

def test_get_args():
    ''' test that get_args correctly parses args
    '''
    res = survivalstan.formulas._get_args('function_name(arg1=1, arg2="2")')
    eq_(res, {'arg1': 1, 'arg2': '2'})
    res = survivalstan.formulas._get_args('function_name(arg1=1, arg2 = "2")')
    eq_(res, {'arg1': 1, 'arg2': '2'})

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
    ''' helper function to determine if set of list elements
        form a sequential set
    '''
    it = (int(el) for el in sorted(set(x)))
    first = next(it)
    return all(a == b for a, b in enumerate(it, first + 1))

def test_is_sequential():
    ''' test whether is_sequential test logic is correct
    '''
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

def test_surv_df(df=get_test_data()):
    ''' test surv_df: that surv stateful transform accepts time & event values
    '''
    res = surv(time=df['time'], event_status=df['event_value'])
    eq_(res.shape[1], 2)
    eq_(res.shape[0], len(df.index))
    ok_(array_equal(set(res.columns), set(['time', 'event_status'])))  # order doesn't matter
    eq_(np.sum(res['event_status']), np.sum(df['event_value']))

def test_surv_df_with_bool():
    ''' test surv_df2: that surv stateful transform works with boolean event types
    '''
    test_surv_df(get_alt_test_data())

def test_surv_df_subject():
    ''' test that surv stateful transform includes subject id when given
    '''
    df = get_test_data()
    res = surv(time=df['time'], event_status=df['event_value'],
               subject=df['subject_id'])
    eq_(res.shape[1], 3)
    eq_(res.shape[0], len(df.index))
    ok_(array_equal(set(res.columns), set(['timepoint_id', 'event_status', 'subject_id'])))  # order doesn't matter
    check_valid_id(res['subject_id'], ref=df['subject_id'])
    check_valid_id(res['timepoint_id'], ref=df['time'])
    eq_(np.sum(res['event_status']), np.sum(df['event_value']))

def test_surv_df_formula(df=get_test_data()):
    y, X = patsy.dmatrices('surv(time=time, event_status=event_value) ~ X1',
                           data=df)
    res = pd.DataFrame(y)
    eq_(res.shape[1], 2)
    eq_(res.shape[0], len(df.index))
    eq_(np.sum(res[0]), np.sum(df['event_value']))
    eq_(np.sum(res[1]), np.sum(df['time']))

def test_surv_df_formula_with_bool():
    test_surv_df_formula(get_alt_test_data())

def test_surv_df_subject_formula(df=get_test_data()):
    formula = 'surv(time=time, event_status=event_value, subject=subject_id) ~ X1'
    y, X = patsy.dmatrices(formula, data=df)
    res = pd.DataFrame(y)
    # test quality of res
    eq_(res.shape[1], 3)
    eq_(res.shape[0], len(df.index))
    # results should be:
    #  0: event_status
    #  1. timeepoint_id
    #  2: subject_id
    eq_(np.sum(res[0]), np.sum(df['event_value']))
    check_valid_id(res[1], ref=df['time'])
    check_valid_id(res[2], ref=df['subject_id'])
    # test whether class ids are retained when predicting new data
    (y.new, X.new) = patsy.build_design_matrices([y.design_info,
                                                   X.design_info], df.tail(n=50))
    res2 = pd.DataFrame(y.new)
    resm = pd.merge(res, res2, on=[1,2], how='inner')
    ok_(array_equal(resm['0_x'], resm['0_y']))

def test_surv_df_subject_formula_with_bool():
    test_surv_df_subject_formula(get_alt_test_data())

def test_SurvivalFactor_formula():
    # basic SurvivalFactor class
    a = SurvivalFactor(code='surv(time=time, event_status=event_value)')
    ok_(a._class is None)
    # test with generic ModelDesc function
    md = patsy.ModelDesc([patsy.Term([a])],[])
    df = get_test_data()
    y, X = patsy.dmatrices(md, data=df)
    eq_(y.shape[1], 2)
    eq_(y.shape[0], len(df.index))
    ok_(y.design_info.terms[0].factors[0]._is_survival) ## should be True
    ok_(issubclass(y.design_info.terms[0].factors[0]._class, SurvData))
    eq_(y.design_info.terms[0].factors[0]._type, 'wide')

def _dict_keys_include(dict_obj, incl):
    ''' Assert that all keys in `incl` exist in dict_obj.keys()
    '''
    not_incl = [key for key in incl if key not in dict_obj.keys()]
    if len(not_incl)>0:
        print('Keys not found in obj: {}'.format(str(not_incl)))
    ok_(len(not_incl)==0)

def test_SurvivalModelDesc_wide_with_bool():
    test_SurvivalModelDesc_wide(get_alt_test_data())

def test_SurvivalModelDesc_wide(df=get_test_data()):
    formula = 'surv(time=time, event_status=event_value) ~ X1'
    my_formula = SurvivalModelDesc(formula)
    y, X = patsy.dmatrices(my_formula, data=df)
    # inspect data frame
    eq_(y.shape[1], 2)
    eq_(y.shape[0], len(df.index))
    # should only be one LHS term
    eq_(len(y.design_info.terms), 1)
    # should only be one LHS factor
    eq_(len(y.design_info.terms[0].factors), 1)
    # LHS should be of type 'survival' (wide)
    ok_(y.design_info.terms[0].factors[0]._is_survival == True)
    ok_(issubclass(y.design_info.terms[0].factors[0]._class, SurvData))
    eq_(y.design_info.terms[0].factors[0]._type, 'wide')
    # stan_data & meta-data should be empty
    _dict_keys_include(y.design_info.terms[0].factors[0]._stan_data,
                       incl=['event', 'y', 'N'])
    _dict_keys_include(y.design_info.terms[0].factors[0]._meta_data,
                       incl=['df'])

def test_SurvivalModelDesc_long_with_bool():
    test_SurvivalModelDesc_long(get_alt_test_data())

def test_SurvivalModelDesc_long(df=get_test_data()):
    formula = 'surv(time=time, event_status=event_value, subject=subject_id) ~ X1'
    my_formula = SurvivalModelDesc(formula)
    y, X = patsy.dmatrices(my_formula, data=df)
    # confirm shape of data returned
    eq_(y.shape[1], 3)
    eq_(y.shape[0], len(df.index))
    ok_(y.design_info.terms[0].factors[0]._is_survival) ## should be True
    eq_(y.design_info.terms[0].factors[0]._class, LongSurvData)
    eq_(y.design_info.terms[0].factors[0]._type, 'long')
    # look for stan_data
    stan_data = y.design_info.terms[0].factors[0]._stan_data
    _dict_keys_include(stan_data,
                        ['t_obs','t_dur','T','S','N','event','t','s'])
    # look for meta-data
    meta_data = y.design_info.terms[0].factors[0]._meta_data
    _dict_keys_include(meta_data, ['timepoint_id', 'subject_id'])
    # can we extract meta-data?
    eq_(meta_data['subject_id'].shape[1], 2)
    eq_(meta_data['subject_id'].shape[0], 100) ## because simulated N=100
    # test ability to build design matrices on new data
    y.new, X.new = patsy.build_design_matrices(design_infos=[y.design_info,
                                                              X.design_info],
                                                data=df.tail(n=50))
    res1 = pd.DataFrame(y)
    res2 = pd.DataFrame(y.new)
    resm = pd.merge(res1, res2, on=[1, 2], how='inner')
    ok_(array_equal(resm['0_x'], resm['0_y']))

def test_SurvivalModelDesc_long_with_group():
    df = get_test_data()
    formula = 'surv(time=time, event_status=event_value, subject=subject_id, group=subject_id) ~ X1'
    my_formula = SurvivalModelDesc(formula)
    y, X = patsy.dmatrices(my_formula, data=df)
    eq_(y.shape[1], 4)
    eq_(y.shape[0], len(df.index))
    ok_(y.design_info.terms[0].factors[0]._is_survival)
    eq_(y.design_info.terms[0].factors[0]._class, LongSurvData)
    eq_(y.design_info.terms[0].factors[0]._type, 'long')
    stan_data = y.design_info.terms[0].factors[0]._stan_data
    _dict_keys_include(stan_data,
                       ['t_obs', 't_dur', 'T', 'S', 'G', 'N', 'event',
                        't', 's', 'g'])
    meta_data = y.design_info.terms[0].factors[0]._meta_data
    _dict_keys_include(meta_data,
                       ['timepoint_id', 'subject_id', 'group_id'])



