import pkg_resources, os

resource_package = __name__  ## Could be any module/package name.
pem_survival_path = os.path.join('stan', 'pem_survival_model.stan')
weibull_survival_path = os.path.join('stan', 'weibull_survival_model.stan')
pem_survival_model = pkg_resources.resource_string(resource_package, pem_survival_path)
weibull_survival_model = pkg_resources.resource_string(resource_package, weibull_survival_path)
