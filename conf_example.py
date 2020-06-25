from configuration import conf

"""
Following properties can be changed:
	- num_distr
	- layer_type
	- model_type
	- hidden_units
"""
conf.model_type, conf.hidden_units = 'CNN', '100'

print(conf.num_distr)
conf.num_distr = '10'
print(conf.num_distr)
