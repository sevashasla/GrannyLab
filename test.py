import pytest
import grannylab as gr
import numpy as np


def equal(arr_left, arr_right):
	return np.sum(np.abs(arr_left - arr_right)) < gr.__eps	


def test_grad_null_error_one():
	a = gr.array([
		12.22
	])
	b = gr.array([
		-39.001
	])
	c = gr.array([
		-46.258
	])
	
	f = a + b + c - a / b - b / c + -c / a + a * b * c
	f.backward()
	a_grad = 1 - 1 / b + c / a ** 2 + b * c
	b_grad = 1 - 1 / c + a / b ** 2 + a * c
	c_grad = 1 - 1 / a + b / c ** 2 + a * b
	assert equal(a.grad, a_grad.values)
	assert equal(b.grad, b_grad.values)
	assert equal(c.grad, c_grad.values)


def test_grad_error():
	a = gr.array([
		[1515.22, 3.1]
	])
	b = gr.array([
		[12233.001, 1.3]
	])
	c = gr.array([
		[-1298.258, 0.5]
	])
	f = a + b - a / b - b / c + a * b * c + a * b + a * c
	f.backward()
	a_grad = b * c + b - 1 / b + c + 1
	b_grad = a / b**2 + a * c + a + 1 - 1 / c
	c_grad = a * b + a + b / c ** 2
	assert equal(a.grad, a_grad.values)
	assert equal(b.grad, b_grad.values)
	assert equal(c.grad, c_grad.values)


def test_grad_more_values():
	a = gr.array([
		[1515.22, 3.1],
		[92.1, 3.0]
	])
	b = gr.array([
		[-273.001, 1.3],
		[1222.001, 0.15]
	])
	c = gr.array([
		[-1298.258, 0.5],
		[-0.218, 0.5]
	])

	f = a ** 2 + b ** 3 + 3 * c - 5 / (a * b)
	f.backward()
	a_grad = 5 / (a ** 2 * b) + 2 * a
	b_grad = 5/(a * b ** 2) + 3 * b ** 2
	c_grad = 3 * np.ones_like(c.values)

	assert equal(a.grad, a_grad.values)
	assert equal(b.grad, b_grad.values)
	assert equal(c.grad, c_grad)


def test_cycle():
	b = gr.array([12.0])
	x = b
	for i in range(10):
		x = x + x

	x.backward()
	print(b.grad)
	assert equal(b.grad, (2 ** 10))


def test_complex_chain():
	a = gr.array([413.2])
	b = gr.array([-0.17])
	c = b * a * b
	k = c + 5 + a
	e = k / c * a
	e.backward()
	
	# e = k / c * a = (c + 5 + a) / c * a = (a * b ** 2 + 5 + a) / (a * b ** 2) * a
	# de/da = 1 / b**2 + 1
	# de/db = -2 * (a + 5) / b**3

	a_grad = 1 / b.values**2 + 1
	b_grad = -2 * (a.values + 5) / b.values**3

	assert equal(a.grad, a_grad)
	assert equal(b.grad, b_grad)

def test_errors():
	a = gr.array([
		[1515.22, 3.1],
		[92.1, 3.0]
	])
	b = gr.array([
		[-273.001, 1.3],
		[1222.001, 0.15]
	])
	c = gr.array([
		[-1298.258, 0.5],
		[-0.218, 0.5]
	])

	f = a ** 2 + b ** 3 + 3 * c - 5 / (a * b)
	f.backward()

	errors = np.sqrt(
		(a.grad * a.errors) ** 2 + (b.grad * b.errors) ** 2 + (c.grad * c.errors) ** 2
	)
	f.count_errors()
	assert equal(f.errors, errors)


def test_errors_complex_chain():
	a = gr.array([413.2])
	b = gr.array([-0.17])
	c = b * a * b
	k = c + 5 + a
	e = k / c * a
	e.backward()
	errors = np.sqrt(
		(a.grad * a.errors) ** 2 + (b.grad * b.errors) ** 2
	)
	e.count_errors()
	assert equal(e.errors, errors)


def test_errors_cycle():
	b = gr.array([12.0])
	x = b
	for i in range(10):
		x = x + x
	x.backward()
	errors = np.sqrt(
		(b.grad * b.errors) ** 2
	)
	x.count_errors()
	assert equal(x.errors, errors)


def test_lstsq():
	x = gr.array([1, 2, 3, 4])
	y = gr.array([1, 2, 3, 4]) * 3 + 5
	k, c = gr.lstsq(x, y)
	assert equal(k.values, 3)
	assert equal(c.values, 5)
