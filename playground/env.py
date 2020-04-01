import numpy as np

class Environment(object):
	"""docstring for Environment"""
	def __init__(self):
		super(Environment, self).__init__()


	def load_data(self):
		# 4 Quadrants
		x = np.random.uniform(-1,1,10000)
		y = np.random.uniform(-1,1,10000)
		x = x[x!=0]
		y = y[y!=0]
		label = x*y > 0

		return np.stack([x,y],1), label
