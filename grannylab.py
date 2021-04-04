from collections.abc import Container, Iterable, Iterator
import math
from numbers import Real
from copy import copy

def sin(x):
	newone = copy(x)
	if isinstance(newone, Iterable):
		for el in newone:
			if isinstance(el, Element):
				el.call_log()
			else:
				el = math.log(el)
	elif isinstance(newone, Element):
		newone.call_log()
	else:
		newone = math.log(newone)


def log(x):
	newone = copy(x)
	if isinstance(newone, Iterable):
		for el in newone:
			if isinstance(el, Element):
				el.call_log()
			else:
				el = math.log(el)
	elif isinstance(newone, Element):
		newone.call_log()
	else:
		newone = math.log(newone)


class ArrayIterator(Iterator):
	def __init__(self, array):
		super().__init__()
		self.array = array
		self.index = 0
	def __iter__(self):
		return self
	def __next__(self):
		self.index += 1
		if self.index <= len(self.array):
			return self.array[self.index - 1]
		raise StopIteration

class Array(Iterable):
	def __init__(self, elements=None):
		super().__init__()
		if(elements is None):
			elements = []
		if isinstance(elements, Real):
			elements = [elements]
		self.elements = list(elements)
		for index in range(len(elements)):
			self.elements[index] = Element(self.elements[index])

	def make_errors_equal(self, error):
		for el in self:
			el.error = error

	def get_errors(self):
		errors = [el.error for el in self.elements]
		return errors

	def get_values(self):
		values = [el.value for el in self.elements]
		return values

	def mean_value(self):
		mean_value = 0.0
		for el in self:
			mean_value += el.value
		mean_value /= len(self)

	def mean(self):
		mean = Element(0.0)
		mean.value = self.mean_value()
		mean.error = math.sqrt(self.mean_error() ** 2.0 + (self.std() / len(self)) ** 2.0)
		return mean

	def std(self):
		std = 0.0
		mean_value = self.mean_value()
		for el in self:
			std += (el.value - mean_value) ** 2.0
		std = math.sqrt(std / len(self))
		return std

	def mean_error(self):
		mean_error = 0.0
		for el in self:
			mean_error += el.error
		mean_error /= len(self)

	def __getitem__(self, key):
		return self.elements[key]

	def __setitem__(self, key, value):
		self.elements[key] = Element(value)

	def __delitem__(self, key):
		del self.elements[key]

	def __len__(self):
		return len(self.elements)

	def __iadd__(self, other):
		if isinstance(other, Array):
			if(len(other) != len(self)):
				raise ValueError("operands have to equal len")
			for index in range(len(self)):
				self[index] += other[index]
		elif isinstance(other, Element):
			for index in range(len(self)):
				self[index] += other
		elif isinstance(other, Real):
			other = Element(other)
			for index in range(len(self)):
				self[index] += other
		else:
			raise TypeError("wrong type")
		return self

	def __isub__(self, other):
		if isinstance(other, Array):
			if(len(other) != len(self)):
				raise ValueError("operands have to equal len")
			for index in range(len(self)):
				self[index] -= other[index]
		elif isinstance(other, Element):
			for index in range(len(self)):
				self[index] -= other
		elif isinstance(other, Real):
			other = Element(other)
			for index in range(len(self)):
				self[index] -= other
		else:
			raise TypeError("wrong type")
		return self

	def __itruediv__(self, other):
		if isinstance(other, Array):
			if(len(other) != len(self)):
				raise ValueError("operands have to equal len")
			for index in range(len(self)):
				self[index] /= other[index]
		elif isinstance(other, Element):
			for index in range(len(self)):
				self[index] /= other
		elif isinstance(other, Real):
			other = Element(other)
			for index in range(len(self)):
				self[index] /= other
		else:
			raise TypeError("wrong type")
		return self

	def __imul__(self, other):
		if isinstance(other, Array):
			if(len(other) != len(self)):
				raise ValueError("operands have to equal len")
			for index in range(len(self)):
				self[index] *= other[index]
		elif isinstance(other, Element):
			for index in range(len(self)):
				self[index] *= other
		elif isinstance(other, Real):
			other = Element(other)
			for index in range(len(self)):
				self[index] *= other
		else:
			raise TypeError("wrong type")
		return self

	def __ipow__(self, power):
		if not isinstance(power, Real):
			raise TypeError("wrong type")
		for index in range(len(self)):
			self[index] **= power
		return self

	def __add__(self, other):
		newone = copy(self)
		newone += other
		return newone

	def __sub__(self, other):
		newone = copy(self)
		newone -= other
		return newone
	
	def __mul__(self, other):
		newone = copy(self)
		newone *= other
		return newone

	def __truediv__(self, other):
		newone = copy(self)
		newone /= other
		return newone

	def __pow__(self, power):
		newone = copy(self)
		newone **= power
		return newone

	def __str__(self):
		return str([el.to_tuple() for el in self])

	def __iter__(self):
		return ArrayIterator(self)


class Element():
	def __init__(self, value: float, error: float=0.0):
		self.value, self.error = self.parameters_helper(value, error)

	def parameters_helper(self, value, error):
		if isinstance(value, Real):
			return float(value), float(error)
		elif isinstance(value, Element):
			return value.value, value.error
		else:
			raise TypeError("wrong type")

	def to_tuple(self):
		return (self.value, self.error)

	def __iadd__(self, other):
		other = Element(other)
		self.value = self.value + other.value
		self.error = math.sqrt(self.error ** 2.0 + other.error ** 2.0)
		return self

	def __isub__(self, other):
		other = Element(other)
		self.value = self.value - other.value
		self.error = math.sqrt(self.error ** 2.0 + other.error ** 2.0)
		return self

	def __imul__(self, other):
		other = Element(other)
		self.value = self.value * other.value
		self.error = math.sqrt((self.value * other.error) ** 2.0 + (self.error * other.value)** 2.0)
		return self


	def __itruediv__(self, other):
		other = Element(other)
		self.value = self.value / other.value
		self.error = math.sqrt((self.error / other.value) ** 2.0 + (self.value / other.value ** 2.0 * other.error) ** 2.0)
		return self

	def __ipow__(self, power):
		self.value = self.value ** power,
		self.error = power * self.value ** (power - 1) * self.error
		return self

	def __add__(self, other):
		newone = copy(self)
		newone += other
		return newone

	def __sub__(self, other):
		newone = copy(self)
		newone -= other
		return newone
	
	def __mul__(self, other):
		newone = copy(self)
		newone *= other
		return newone

	def __truediv__(self, other):
		newone = copy(self)
		newone /= other
		return newone

	def __pow__(self, power):
		newone = copy(self)
		newone **= power
		return newone

	def __neg__(self):
		return Element(-self.value, self.error)

	def __pos__(self):
		return Element(+self.value, self.error)

	def __abs__(self):
		return Element(abs(self.value), self.error)

	def __int__(self):
		return int(self.value)

	def __float__(self):
		return self.value

	def __str__(self):
		value = self.value
		error_str = list(str(self.error))
		error = self.error

		first_nonull = len(error_str)
		index_dot = len(error_str)
		for index, sym in enumerate(error_str):
			if(sym.isdigit() and int(sym) != 0):
				first_nonull = index
				break
			if(sym == '.'):
				index_dot = index
		
		if first_nonull != len(error_str):
			count = 0 if first_nonull < index_dot else first_nonull - index_dot
			try:
				supplement = 1 if error_str[first_nonull + 1] != 0 else 0
			except IndexError:
				supplement = 0
				pass

			error = (int(error * 10 ** count) + supplement) / (10 ** count)
			value = round(value, count)

		return "value: {}, error: {}".format(value, error)

	def __repl__(self):
		return "value: {}, error: {}".format(self.value, self.error)

	def __lt__(self, other):
		other = Element(other)
		return (self.value < other.value)

	def __gt__(self, other):
		return (other < self)

	def call_sin(self):
		self.value = math.sin(self.value)
		self.error = math.cos(self.value) * self.error

	def call_log(self):
		self.value = math.log(self.value)
		self.error = self.error	/ self.value

	def percentages(self):
		return self.error / self.value * 100.0
