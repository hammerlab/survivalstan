
import statsmodels
import stanmodels

def load_test_dataset():
	''' Load test dataset from R survival package
	'''
	dataset = statsmodels.datasets.get_rdataset(package = 'survival', dataname = 'flchain' )
	d  = dataset.data.query('futime > 7')
	d.reset_index(level = 0, inplace = True)
	return(d)

def test_weibull_model():
	''' Test Weibull survival model on test dataset
	'''
	d = load_test_dataset()
	testfit = stanmodels.fit_stan_survival_model(
		model_cohort = 'test model',
		model_code = stanmodels.stan.weibull_survival_model,
		df = d,
		time_col = 'futime',
		event_col = 'death',
		formula = 'age + sex',
		iter = 3000,
		chains = 4,
		make_inits = stanmodels.make_weibull_survival_model_inits
		)
	return(testfit)

def test_pem_model():
	''' Test Weibull survival model on test dataset
	'''
	d = load_test_dataset()
	dlong = stanmodels.prep_data_long_surv(d, time_col = 'futime', event_col = 'death')
	testfit = stanmodels.fit_stan_survival_model(
		model_cohort = 'test model',
		model_code = stanmodels.stan.pem_survival_model,
		df = dlong,
		sample_col = 'index',
		timepoint_col = 'end_time',
		event_col = 'end_failure',
		formula = 'age + sex',
		iter = 3000,
		chains = 4,
		)
	return(testfit)

