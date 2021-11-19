# Welcome to GrannyLab
This project helps students to process their measurements.

### Just paste in terminal
```bash
$ pip install --extra-index-url https://testpypi.python.org/pypi GrannyLab
```
### How to use it?
`array` is a class, which can make automatic caltucation of errors (*standard deviation*).
Each `array` consist fields: **values** and **errors**. There are some ways to create it:
```python
import GrannyLab as gr
# from single array
x = gr.array([
	[1.0, 2.0], [3.0, 4.0],
])
# x.values = [1.0, 3.0], x.errors = [2.0, 4.0]

# just values, by default errors will be equal to zero
x = gr.array([1.0, 2.0])
# x.values = [1.0, 2.0], x.errors = [0.0, 0.0]

# two arrays: one for values and another for errors
x = gr.array(
	[1.0, 2.0], 
	[3.0, 4.0]
)
# x.values = [1.0, 2.0], x.errors = [3.0, 4.0]

# array and number, array for values and 
# all errors will be equal to number
x = gr.array(
	[1.0, 2.0], 3.0
)
x.values = [1.0, 2.0], x.errors = [3.0, 3.0]
```
### Array support such operations
```python
import GrannyLab as gr
# let x, y - arrays with equal sizes
# c - number
x + y, x - y, x / y, x * y
x + c, x - c, x * c, x / c, x ** c
gr.log(x), gr.exp(x), gr.sqrt(x)
```
### Count errors
After all calculations one should call `.count_errors()` method and array automatically calculate errors from leafs to root. Let's look at an example:
```python
import GrannyLab as gr
x = gr.array([1.0, -2.0], [1.0, 0.1])
y = gr.array([2.0, -3.0], [0.1, 0.15])
z = x + y / x # one can use more complex function
z.count_errors()
print(z.errors)
```
During the computations `array` saves the graph of computation, than it uses chain rule to push derivatives to leafs and than it count errors. Like this:
```
	 z
     / \
    /   \
   x   (x / y)
   		/  \
	   x    y
```
### The Least Squares Method
`lstsq(x, y, coeff_is_null=False)` is a function which can find the solution for equation `y = k * x + b` via [method of the least squares](https://en.wikipedia.org/wiki/Least_squares). If one knows that b = 0 than he should set the parameter coeff_is_null = True.
```python
import GrannyLab as gr
x = gr.array([1, 2, 3, 4])
y = gr.array([1, 2, 3, 4]) * 3 + 5
k, b = gr.lstsq(x, y)
```

### experimental:
There is a method `.get(key)` which returns `array` from a single element (*value and error*) and one can make arithmetic operations on it. But it doesn't work good enough... So use ordinary `__getitem__(key)` method.

In the future I want to make method which convert graph into one latex equation.

### bugs
If you see bugs, please, text me on [vk](https://vk.com/sevashasla) or on telegram @sevashasla