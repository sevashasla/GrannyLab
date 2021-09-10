# made by sevashasla
import numpy as np
import math
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from numbers import Number


pi = math.pi
e = math.e
_eps = 1e-7


class array():
	def __init__(self, *args, **kwargs):

		if len(args) == 1:
			arg = np.array(args[0], dtype=np.float32)
			if(len(arg.shape) == 2):  # values and errors
				self.values = arg[:, 0].copy()
				self.errors = arg[:, 1].copy()

			else:  # only values
				self.values = arg.copy()
				self.errors = np.zeros_like(self.values, dtype=np.float32)

		else:
			self.values = np.array(args[0], dtype=np.float32)
			self.errors = np.array(args[1], dtype=np.float32)

		self.grad = np.zeros_like(self.values, dtype=np.float32)
		self.grad_layer = None
		self.is_leaf = True
		self.dependencies = None  # Do I need this?

	def create_from_number(self, maybe_number):
		'''
				only two versions: array or number
		'''
		if isinstance(maybe_number, Number):
			maybe_number = array(
				np.full_like(self.values, maybe_number, dtype=np.float32)
			)

		return maybe_number

	def __add__(self, other):
		layer = AddLayer()
		other = self.create_from_number(other)
		return layer(self, other)

	# other + self
	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		layer = SubLayer()
		other = self.create_from_number(other)
		return layer(self, other)

	# other - self
	def __rsub__(self, other):
		return self * (-1) + other

	def __mul__(self, other):
		layer = MultiplyLayer()
		other = self.create_from_number(other)
		return layer(self, other)

	# other * self
	def __rmul__(self, other):
		return self * other

	def __pow__(self, deg):
		layer = DegreeLayer()
		return layer(self, deg)

	def __truediv__(self, other):
		layer = DivideLayer()
		other = self.create_from_number(other)
		return layer(self, other)

	# other / self
	def __rtruediv__(self, other):
		return self ** (-1) * other

	def backward(self, upper=None):
		if upper is None:
			self.grad = np.ones_like(self.values)
		else:
			self.grad += upper

		if not self.is_leaf:
			self.grad_layer.backward_layer(self.grad)

	def zero_grad(self):
		self.grad = np.zeros_like(self.grad)
		if not self.is_leaf:
			self.grad_layer.zero_grad_layer()

	def count_errors(self, arr_result=None):
		if arr_result is None:
			self.grad_layer.count_errors(self.errors)
			self.errors = np.sqrt(self.errors)
		else:
			if not self.is_leaf:
				self.grad_layer.count_errors(arr_result)
			else:
				arr_result += (self.grad * self.errors) ** 2.0

########################################################################
########################################################################
#            I also have to add here code of Layers
########################################################################
########################################################################


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


class AddLayer(Layer):
	'''
			f(z), where z = x + y
			I have df/dz, so
			df/dx = df/dz * dz/dx = df/dz
			df/dy = df/dz * dz/dy = df/dz
	'''

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
		self.arr_right.backward(other_grad)
		self.arr_left.backward(other_grad)

	def zero_grad_layer(self):
		self.arr_left.zero_grad()
		self.arr_right.zero_grad()

	def count_errors(self, arr_result):
		self.arr_left.count_errors(arr_result)
		self.arr_left.zero_grad()
		self.arr_right.count_errors(arr_result)
		self.arr_right.zero_grad()


class SubLayer(Layer):
	'''
			f(z), where z = x - y
			I have df/dz, so
			df/dx = df/dz * dz/dx = df/dz
			df/dy = df/dz * dz/dy = -df/dz
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, arr_left, arr_right):
		self.arr_right = arr_right
		self.arr_left = arr_left

		result = array(arr_left.values - arr_right.values)
		result.grad_layer = self
		result.is_leaf = False

		return result

	def backward_layer(self, other_grad):
		self.arr_right.backward(-other_grad)
		self.arr_left.backward(other_grad)

	def zero_grad_layer(self):
		self.arr_left.zero_grad()
		self.arr_right.zero_grad()

	def count_errors(self, arr_result):
		self.arr_left.count_errors(arr_result)
		self.arr_left.zero_grad()
		self.arr_right.count_errors(arr_result)
		self.arr_right.zero_grad()


class MultiplyLayer(Layer):
	'''
			f(z), where z = x * y
			I have df/dz, so
			df/dx = df/dz * dz/dx = df/dz * y
			df/dx = df/dz * dz/dy = df/dz * x
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, arr_left, arr_right):
		self.arr_left = arr_left
		self.arr_right = arr_right

		result = array(arr_left.values * arr_right.values)
		result.grad_layer = self
		result.is_leaf = False

		return result

	def backward_layer(self, other_grad):
		self.arr_left.backward(other_grad * self.arr_right.values)
		self.arr_right.backward(other_grad * self.arr_left.values)

	def zero_grad_layer(self):
		self.arr_left.zero_grad()
		self.arr_right.zero_grad()

	def count_errors(self, arr_result):
		self.arr_left.count_errors(arr_result)
		self.arr_left.zero_grad()
		self.arr_right.count_errors(arr_result)
		self.arr_right.zero_grad()


class DegreeLayer(Layer):
	'''
			f(z), where z = x^deg
			I have df/dz, so
			df/dx = df/dz * dz/dx = df/dz * deg * x^(deg - 1)
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, arr, deg):
		self.arr = arr
		self.deg = deg

		result = array(arr.values ** deg)
		result.grad_layer = self
		result.is_leaf = False

		return result

	def backward_layer(self, other_grad):
		push_grad = other_grad * self.deg * self.arr.values ** (self.deg - 1)
		self.arr.backward(push_grad)

	def zero_grad_layer(self):
		self.arr.zero_grad()

	def count_errors(self, arr_result):
		self.arr.count_errors(arr_result)
		self.arr.zero_grad()


class DivideLayer(Layer):  # maybe it will be faster than x * y^{-1}
	'''
			f(z), where z = x / y
			I have df/dz, so
			df/dx = df/dz * dz/dx = df/dz * 1 / y
			df/dx = df/dz * dz/dy = -df/dz * x / y ** 2
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, arr_left, arr_right):
		self.arr_left = arr_left
		self.arr_right = arr_right

		result = array(arr_left.values / arr_right.values)
		result.grad_layer = self
		result.is_leaf = False

		return result

	def backward_layer(self, other_grad):
		self.arr_left.backward(other_grad / self.arr_right.values)
		self.arr_right.backward(-other_grad * self.arr_left.values /
								self.arr_right.values ** 2)

	def zero_grad_layer(self):
		self.arr_left.zero_grad()
		self.arr_right.zero_grad()

	def count_errors(self, arr_result):
		self.arr_left.count_errors(arr_result)
		self.arr_left.zero_grad()
		self.arr_right.count_errors(arr_result)
		self.arr_right.zero_grad()


def exp(x):
	layer = ExpLayer()
	return layer(x)


def ExpLayer():
	'''
			f(z), where z = exp(x)
			I have df/dz, so
			df/dx = df/dz * dz/dx = df/dz * exp(x)
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, arr, deg):
		self.arr = arr

		result = array(np.exp(arr.values))
		result.grad_layer = self
		result.is_leaf = False

		return result

	def backward_layer(self, other_grad):
		push_grad = other_grad * np.exp(self.arr)
		self.arr.backward(push_grad)

	def zero_grad_layer(self):
		self.arr.zero_grad()

	def count_errors(self, arr_result):
		self.arr.count_errors(arr_result)
		self.arr.zero_grad()


def log(x):
	layer = LogLayer()
	return layer(x)


class LogLayer():
	'''
			f(z), where z = ln(x)
			I have df/dz, so
			df/dx = df/dz * dz/dx = df/dz * 1/x
	'''

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, arr, deg):
		self.arr = arr

		result = array(np.log(arr.values))
		result.grad_layer = self
		result.is_leaf = False

		return result

	def backward_layer(self, other_grad):
		push_grad = other_grad / self.arr
		self.arr.backward(push_grad)

	def zero_grad_layer(self):
		self.arr.zero_grad()

	def count_errors(self, arr_result):
		self.arr.count_errors(arr_result)
		self.arr.zero_grad()
