from configuration import conf

"""
Following properties can be changed:
	- num_distr
	- layer_type
	- model_type
	- hidden_units
"""

print(conf.num_distr)
conf.num_distr = '10'
print(conf.num_distr)