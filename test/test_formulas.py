from survivalstan import formulas
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
    # should have 3 columns
    eq_(res.shape[1], 3)
    # should be numeric type
    ok_(res[1].dtype == np.float64)
    ok_(res[2].dtype == np.float64)
    # should have min value of 1
    eq_(np.min(res[1]), 1)
    eq_(np.min(res[2]), 1)
    # should have max value equal to number of unique values
    eq_(np.max(res[1]), len(df['time'].unique()))
    eq_(np.max(res[2]), len(df['subject_id'].unique()))
    # TODO test one-to-one & onto relationship



