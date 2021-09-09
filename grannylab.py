import numpy as np
import math
from copy import copy, deepcopy
from abc import ABC, abstractmethod


pi = math.pi
e = math.e
_eps = 1e-7


# @property
class array():
	def __init__(self, *args, **kwargs):

		if len(args) == 1:
			arg = np.array(args[0])
			if(arg.shape[0] == 2): # values and errors
				self.values = arg[:, 0].copy()
				self.errors = arg[:, 1].copy()

			else: # only values
				self.values = arg.copy()
				self.erorrs = np.zeros_like(self.values)

		else:
			self.values = np.array(args[0])
			self.errors = np.array(args[1])

		self.grad = np.zeros_like(self.values)
		self.grad_layer = None
		self.is_leaf = True

	def __add__(self, other):
		layer = Summator()
		return layer(self, other)

	def __sub__(self, other):
		layer = Summator()
		return layer(self, other)

	def backward(self, upper=None):
		if upper is None:
			raise NotImplementedError()
		else:
			self.grad += upper
			self.grad_layer.backward_layer(self.grad)


	def zero_grad(self):
		self.grad = np.zeros_like(self.grad)
		if not self.grad_layer is None:
			self.grad_layer.zero_grad_layer()





class Layer(ABC):
	def __init__(self, *args, **kwargs):
		pass

	@abstractmethod
	def backward_layer(self, other_grad):
		raise NotImplementedError()

	@abstractmethod
	def zero_grad_layer(self):
		raise NotImplementedError()

	@abstractmethod
	def count_errors(self):
		raise NotImplementedError()		


class Summator(Layer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, arr_left, arr_right):
		self.arr_right = arr_right
		self.arr_left = arr_left

		result = array(arr_left.values + arr_right.values)
		result.grad_layer = self
		result.is_leaf = False

		return result


	def backward_layer(self, other_grad):
		raise NotImplementedError()

	def zero_grad_layer(self):
		arr_left.zero_grad()
		arr_right.zero_grad()

	
	


class Subtractor(Layer):
	pass

