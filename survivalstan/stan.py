import pkg_resources, os

resource_package = __name__  ## Could be any module/package name.

_pem_survival_path = os.path.join('stan', 'pem_survival_model.stan')
_pem_survival_varcoef_path = os.path.join('stan', 'pem_survival_model_varying_coefs.stan')
_pem_survival_varcoef_path2 = os.path.join('stan', 'pem_survival_model_varying_coefs2.stan')
_pem_survival_varcoef_path4 = os.path.join('stan', 'pem_survival_model_varying_coefs4.stan')
_weibull_survival_path = os.path.join('stan', 'weibull_survival_model.stan')
_weibull_survival_varcoef_path = os.path.join('stan', 'weibull_survival_model_varying_coefs.stan')

pem_survival_model = pkg_resources.resource_string(
	resource_package, _pem_survival_path).decode("utf-8") 
pem_survival_model_varying_coefs = pkg_resources.resource_string(
	resource_package, _pem_survival_varcoef_path).decode("utf-8") 
pem_survival_model_varying_coefs2 = pkg_resources.resource_string(
	resource_package, _pem_survival_varcoef_path2).decode("utf-8") 
pem_survival_model_varying_coefs4 = pkg_resources.resource_string(
	resource_package, _pem_survival_varcoef_path4).decode("utf-8") 
weibull_survival_model = pkg_resources.resource_string(
	resource_package, _weibull_survival_path).decode("utf-8") 
weibull_survival_model_varying_coefs = pkg_resources.resource_string(
	resource_package, _weibull_survival_varcoef_path).decode("utf-8") 
