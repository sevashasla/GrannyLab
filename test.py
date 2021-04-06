import pytest
import grannylab as gr
from random import randint
from math import sin, cos, log


def equal(x, y):
	eps = 1e-4
	return abs(x - y) < eps

def test_add():
	a = gr.Element(1.0, 123.0)
	b = gr.Element(-14.0, 12.0)
	c = gr.Element(0.0, 1.0)
	d = gr.Element(-500.0, 1.0)
	a += b
	assert equal(a.error -123.58397954427588, 0.0)
	assert equal((b + c).value - -14.0, 0.0)
	assert equal((c + d).error, 1.4142135623730951)
	c += (b + d)
	assert equal(c, -514)

def test_add_array():
	a = gr.Array([gr.Element(1.0, 123.0), gr.Element(1.0, 123.0)])
	b = gr.Array([gr.Element(-14.0, 12.0), gr.Element(-14.0, 12.0)])
	c = gr.Array([gr.Element(0.0, 1.0), gr.Element(0.0, 1.0)])
	d = gr.Array([gr.Element(-500.0, 1.0)])
	a += b
	assert equal(a[0].error -123.58397954427588, 0.0)
	assert equal((b + c)[0].value - -14.0, 0.0)
	assert equal((c + d)[0].error, 1.4142135623730951)
	c += (b + d)
	assert equal(c[0], -514)

def test_sub():
	a = gr.Element(13.0, 0.012)
	b = gr.Element(0.0, 1.0)
	c = gr.Element(-500.0, 1.0)
	assert equal((a - b).value, 13.0)
	a += b
	assert equal(a.error, 1.0000719974081866)
	b += c
	assert equal(b.error, 1.4142135623730951)
	assert equal((c + 500.0).value, 0)


def test_mul():
	a = gr.Element(1.0, 0.15)
	b = gr.Element(-3.0, 1.0)
	c = gr.Element(14.0, 0.1)
	assert equal((a * b).value, -3.0)
	assert equal((c * 0.5).value, 7.0)
	assert equal((c * 0.5).error, 0.05)
	b *= c
	assert equal(b.error, 14.003213916812097)


def test_mul_array():
	a = gr.Array([gr.Element(1.0, 0.15)])
	b = gr.Array([gr.Element(-3.0, 1.0)])
	c = gr.Array([gr.Element(14.0, 0.1)])
	assert equal((a * b)[0].value, -3.0)
	assert equal((c * 0.5)[0].value, 7.0)
	assert equal((c * 0.5)[0].error, 0.05)
	b *= c
	assert equal(b[0].error, 14.003213916812097)

def test_div():
	a = gr.Element(1.0, 0.15)
	b = gr.Element(-3.0, 0.78)
	assert equal((a / b).value, -1/3)
	assert equal((a / b).error, 0.10005554013202422) 
	b /= 3
	assert equal(b.error, 0.26)
	b /= a
	assert equal(b.error, 0.3001666203960727)

def test_pow():
	a = gr.Element(1.0, 0.15)
	b = gr.Element(-2.0, 9.0)
	assert equal((a ** 5.0).error, 0.75)
	assert equal((a ** 5.0).value, 1.0)
	assert equal((b ** 10).error, 46080.0)
	assert equal((b ** 10).value, 1024.0)

def test_formula():
	x = gr.Element(1.0, 0.29)
	y = gr.Element(5.0, 0.31)
	z = gr.Element(0.12, 0.056)

	f = lambda x, y, z: x * y / z * 5- z / y + x ** 5
	res = gr.formula(f, x, y, z)
	assert equal(res.value, f(x, y, z).value)
	print(res.error)
	assert equal(res.error, 115.9685441)

def test_array_formula():
	x = gr.Array([gr.Element(1.0, 0.29), gr.Element(1.0, 0.29)])
	y = gr.Array([gr.Element(5.0, 0.31), gr.Element(5.0, 0.31)])
	z = gr.Element(0.12, 0.056)

	f = lambda x, y, z: x * y / z * 5 - z / y + x ** 5

	res = gr.formula_array(f, x, y, z)
	assert equal(res[0].error, 115.9685441)
	assert equal(res[1].value, f(x, y, z)[1].value)
	assert equal(res[1].error, 115.9685441)

def test_big_formula_without_sin_log():
	x = gr.Array([gr.Element(1.0, 0.29), gr.Element(2.0, 0.29), gr.Element(7.08, 0.9), gr.Element(-14.88, 0.3)])
	y = gr.Array([gr.Element(5.0, 0.31), gr.Element(-6.0, 0.31), gr.Element(-5.0, 0.29), gr.Element(13.37, 0.29)])
	z = gr.Element(0.12, 0.14)
	k = gr.Array([1.0, 2.0, -3.0, 4.0])
	w = gr.Element(1.0)

	f = lambda x, y, z, k, w: 3 * x ** 2 + 5 * y * w / 15.0 - 164 * w**5 * x / z ** 2 + 446 * w * x * 1.0 / k + k * w / z
	f(x, y, z, k, w)
	res = gr.formula_array(f, x, y, z, k, w)
	assert equal(res[0].value, f(x, y, z, k, w)[0].value)

def test_sin_and_cos():
	x = gr.Array([gr.Element(gr.pi, 1.0), gr.Element(gr.pi / 2, 0.88), gr.Element(gr.pi / 4, 15.4)])
	x = gr.sin(x)
	for el in x:
		assert equal(gr.sin(el).value, sin(el.value))
		assert equal(gr.cos(el).value, cos(el.value))
		assert equal(gr.sin(el).error, abs(cos(el.value) * el.error))
		assert equal(gr.cos(el).error, abs(sin(el.value) * el.error))


def test_log():
	x = gr.Array([gr.Element(gr.e ** 1.0), gr.Element(gr.e ** 4.0), gr.Element(gr.e ** 5.0)])
	for el in x:
		assert equal(gr.log(el).value, log(el.value))
		assert equal(gr.log(el).error, abs(el.error / el.value))

def test_big_formula_just():
	f = lambda x, y, z, k: x ** 2.0 / (y * gr.log(k) * gr.sin(x)) + 23.0 * gr.sin(k * gr.log(x)) - x / y / z / k + gr.log(x * y * z * k ** 2)
	x = gr.Array([gr.Element(11.0, 0.29), gr.Element(2.0, 0.29), gr.Element(7.08, 0.9), gr.Element(14.88, 0.3)])
	y = gr.Array([gr.Element(5.0, 0.31), gr.Element(6.0, 0.31), gr.Element(5.0, 0.29), gr.Element(13.37, 0.29)])
	z = gr.Element(0.12, 0.14)
	k = gr.Array([16.0, 2.0, 3.0, 4.0])
	assert equal(f(x, y, z, k)[0].value, gr.formula_array(f, x, y, z, k)[0].value)
	assert equal(f(x, y, z, k)[1].value, gr.formula_array(f, x, y, z, k)[1].value)
	assert equal(f(x, y, z, k)[2].value, gr.formula_array(f, x, y, z, k)[2].value)
	assert equal(f(x, y, z, k)[3].value, gr.formula_array(f, x, y, z, k)[3].value)


def test_LeastSquares():
	x = gr.Array([1.0, 2.0, 3.0])
	y = gr.Array([2.0, 5.0, 6.0])
	assert equal(gr.LeastSquares(x, y)[0].value, 2.0)

