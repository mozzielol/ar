import configparser
import numpy as np
"""
	|- Task Setting
		|`- dataset_name
	|- Model Setting:
		|- model_type: Model Type
			|- NN : Neural Networks
			|- CNN(#TODO) : Convolutional Neural Networks

		|- hidden_units: list of model units


	|- Training Setting:
		|- batch_size
		|- num_epoch
		|- learning_rate
		|- optimizer: which optimizer to use

"""

class Config(object):
	def __init__(self, config_file=None, base_path = './configuration/settings/'):
		default_file =  './default.ini'
		self.cfg = configparser.ConfigParser()
		self.cfg.read(default_file)
		if config_file is not None:
			path = base_path + config_file
			self.cfg.read(path)

	def read_list(self,input_str):

		task_labels = []
		count = -1
		for i in input_str:
			if i in [',',' ']:
				pass
			elif i is '|':
				count += 1
				task_labels.append([])
			else:
				task_labels[count].append(int(i))
		return task_labels

	######################## Task Setting ##########################
	@property
	def dataset_name(self):
		return self.cfg.get('Task Setting', 'dataset_name')
	
	
	######################## Model Setting ##########################
	@property
	def model_type(self):
		return self.cfg.get('Model Setting', 'model_type')

	@property
	def hidden_units(self):
		return np.array(self.cfg.get('Model Setting', 'hidden_units').split(',')).astype(np.int32)


	######################## Last Layer Setting ##########################
	@property
	def layer_type(self):
		return self.cfg.get('Last Layer Setting', 'layer_type')

	@property
	def num_distr(self):
		return self.cfg.getint('Last Layer Setting', 'num_distr')
	
	@property
	def output_units(self):
		return self.cfg.getint('Last Layer Setting', 'output_units')

	######################## Training Setting ##########################

	@property
	def batch_size(self):
		return self.cfg.getint('Training Setting', 'batch_size')

	@property
	def num_epoch(self):
		return self.cfg.getint('Training Setting', 'num_epoch')


	@property
	def learning_rate(self):
		return self.cfg.getfloat('Training Setting', 'learning_rate')

	@property
	def optimizer(self):
		return self.cfg.get('Training Setting', 'optimizer')


	@num_distr.setter
	def num_distr(self,value):
		self.cfg.set('Last Layer Setting', 'num_distr', value)

	@layer_type.setter
	def layer_type(self,value):
		self.cfg.set('Last Layer Setting', 'layer_type', value)

	@layer_type.setter
	def layer_type(self,value):
		self.cfg.set('Last Layer Setting', 'layer_type', value)


	@model_type.setter
	def model_type(self,value):
		self.cfg.set('Model Setting', 'model_type', value)

	@hidden_units.setter
	def hidden_units(self,value):
		self.cfg.set('Model Setting', 'hidden_units', value)








	
	
	
	
