# made by sevashasla
from __future__ import annotations
import numpy as np
import math
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from numbers import Number


pi = math.pi
e = math.e
__eps = 1e-7


def lstsq(x: array, y: array, coeff_is_null=False) -> (array, array):
    '''
        This function returns solution for equation
        y = k * x + b
        If one knows that b = 0, than he should set coeff_is_null = True
        example of usage:
            # let x, y - gr.arrays with equal sizes
            lstsq(x, y)
            or if b = 0:
            lstsq(x, y, True)
    '''
    n = len(x)
    if len(x) != len(y):
        raise RuntimeError("x and y must have equal len")

    if coeff_is_null:
        k = np.mean(x * y) / np.mean(x ** 2)
        c = 0.0
        k_err = 1.0 / np.sqrt(n) * np.sqrt(
            abs((np.mean(y.values ** 2) - np.mean(y.values) ** 2) /
                (np.mean(x.values ** 2) - np.mean(x.values) ** 2) - k ** 2)
        )
        c_err = 0.0

    else:
        k = (np.mean(x.values * y.values) \
            - np.mean(x.values) * np.mean(y.values)) \
        / (np.mean(x.values ** 2) \
            - np.mean(x.values) ** 2)

        c = np.mean(y.values) - k * np.mean(x.values)
        k_err = 1.0 / np.sqrt(n) * np.sqrt(abs((np.mean(y.values ** 2)
                                                - np.mean(y.values) ** 2)
                                                / (np.mean(x.values ** 2)
                                                - np.mean(x.values) ** 2) - k ** 2))

        c_err = k_err * np.sqrt(np.mean(x.values ** 2) -
                                np.mean(x.values) ** 2)

    # two critical angles of the line
    k1 = ((y.values[-1] + y.errors[-1]) - (y.values[0] - y.errors[0])) /\
        ((x.values[-1] - x.errors[-1]) - (x.values[0] + x.errors[0]))
    c1 = (y.values[-1] + y.errors[-1]) - k1 * (x.values[-1] - x.errors[-1])

    k2 = ((y.values[-1] - y.errors[-1]) - (y.values[0] + y.errors[0])) /\
        ((x.values[-1] + x.errors[-1]) - (x.values[0] - x.errors[0]))
    c2 = (y.values[-1] - y.errors[-1]) - k1 * (x.values[-1] + x.errors[-1])

    k_err_add = np.sqrt((k2 - k1) ** 2) / np.sqrt(n)
    c_err_add = np.sqrt((c2 - c1) ** 2) / np.sqrt(n)

    k_err = np.sqrt(k_err ** 2.0 + k_err_add ** 2.0)
    if coeff_is_null:
        c_err = np.sqrt(c_err ** 2.0 + c_err_add ** 2.0)

    return array([[k, k_err]]), array([[c, c_err]])


class array():
    def __init__(self, *args, **kwargs) -> None:
        '''
            example of calling:
            1:
                gr.array([
                    [1.0, 2.0], 
                    [3.0, 4.0],
                ])
                in this case 
                values: 1.0, 3.0
                errors: 2.0, 4.0
            2:
                gr.array([1.0, 2.0])
                in this case:
                values: 1.0, 2.0
                errors: 0.0, 0.0
            3:
                gr.array(
                    [1.0, 2.0], 
                    [3.0, 4.0]
                )
                in this case:
                values: 1.0, 2.0
                errors: 3.0, 4.0
            4:
                gr.array(
                    [1.0, 2.0], 3.0
                )
                in this case:
                values: 1.0, 2.0
                errors: 3.0, 3.0
        '''

        if len(args) == 1:
            arg = np.array(args[0], dtype=np.float32)
            if len(arg.shape) == 2:
                # values and errors, but in one array
                self.values = arg[:, 0].copy()
                self.errors = arg[:, 1].copy()

            else:
                # only values
                self.values = arg.copy()
                self.errors = np.zeros_like(self.values, dtype=np.float32)
        else:
            # values and errors, but in two arrays
            self.values = np.array(args[0], dtype=np.float32)
            if isinstance(args[1], Number):
                self.errors = np.full_like(self.values, args[1], np.float32)
            else:
                self.errors = np.array(args[1], dtype=np.float32)

        self.grad = np.zeros_like(self.values, dtype=np.float32)
        self.grad_layer = None
        self.is_leaf = True

    def __getitem__(self, key: int) -> tuple:
        return (self.values[key], self.errors[key])

    def get(self, key: int) -> tuple:
        '''
            returns item via key
        '''
        layer = SelectLayer()
        return layer(self, key)

    def item(self) -> (np.float32, np.float32):
        '''
            This function returns pair from signle-element array
            example of usage:
                x = gr.array([1.0, 0.1])
                print(x.item())

            [!] this case won't work:
                x = gr.array([1.0, 1.0], [0.1, 0.1])
                print(x.item)
        '''

        if len(self.arr.values) == 1:
            return self.values[0], self.errors[0]
        else:
            raise RuntimeError("only for single-element array")

    def mean(self) -> array:
        '''
        This function returns mean with relevant errors
        '''
        return array([[self.values.mean(), np.sqrt(self.errors.mean() /
                                                   len(self) + self.values.std() ** 2)]])

    def __setitem__(self, key: int, value: np.float32):
        if isinstance(value, Number):
            self.values[key] = value
        else:
            self.values[key], self.errors[key] = value

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return "\n".join(["(" + str(self.values[i]) + ", " +
                          str(self.errors[i]) + ")" for i in range(len(self))])

    def __create_from_number(self, maybe_number):
        '''
                If maybe_number is a number than 
                it returns vector full of maybe_number
                with shapes like in current array.

                Otherwise it returns maybe_number
        '''
        if isinstance(maybe_number, Number):
            maybe_number = array(
                np.full_like(self.values, maybe_number, dtype=np.float32)
            )

        return maybe_number

    def __add__(self, other) -> array:
        layer = AddLayer()
        other = self.__create_from_number(other)
        return layer(self, other)

    # other + self
    def __radd__(self, other) -> array:
        return self + other

    def __sub__(self, other) -> array:
        layer = SubLayer()
        other = self.__create_from_number(other)
        return layer(self, other)

    # other - self
    def __rsub__(self, other) -> array:
        return self * (-1) + other

    def __mul__(self, other) -> array:
        layer = MultiplyLayer()
        other = self.__create_from_number(other)
        return layer(self, other)

    # other * self
    def __rmul__(self, other) -> array:
        return self * other

    def __pow__(self, deg: Number) -> array:
        layer = DegreeLayer()
        return layer(self, deg)

    def __truediv__(self, other) -> array:
        layer = DivideLayer()
        other = self.__create_from_number(other)
        return layer(self, other)

    # other / self
    def __rtruediv__(self, other) -> array:
        return self ** (-1) * other

    def __neg__(self) -> array:
        return -1 * self

    def backward(self, upper=None) -> None:
        '''
            This function count gradient via
            chain rule from this point to leafs

            example of usage:
                x = gr.array([1.0, 0.1])
                y = x ** 2
                z = y ** 2 + x
                z.backward()
        '''
        if upper is None:
            self.grad = np.ones_like(self.values)
            upper = self.grad
        else:
            # we should watch if this array becomes from
            # SelectLayer because it has different size
            if isinstance(self.grad_layer, SelectLayer):
                upper = np.array([sum(upper)])

            self.grad += upper

        if not self.is_leaf:
            self.grad_layer.backward_layer(upper)

    def zero_grad(self):
        '''
            It makes gradient equal to zero here and 
            in all dependencies elements.
            example of usage:
                x = gr.array([1.0, 0.1])
                y = x ** 2
                y.backward()
                y.zero_grad()
        '''
        self.grad = np.zeros_like(self.grad)
        if not self.is_leaf:
            self.grad_layer.zero_grad_layer()

    def count_errors(self):
        '''
            This function count errors of a sequence 
            of arithmetic equations

            Example of usage:
                x = gr.array([1.0, 0.1])
                y = x ** 2
                y.count_errors()
                print(y.errors)


            [!] Before calling one should make sure that all
            gradients are equal to zero, if he has called
            backward function
        '''

        self.backward()
        self.count_errors_if_grad_counted()
        # One don't need to call zero_grad due to
        # It have made during count_error_if_grad_counted

    def count_errors_if_grad_counted(self, arr_result=None):
        '''
            It counts errors from leafs, only [!] if 
            gradients have already been counted
        '''
        if arr_result is None:
            self.errors.fill(0)
            self.grad_layer.count_errors(self.errors)
            self.errors = np.sqrt(self.errors)
        else:
            if not self.is_leaf:
                self.grad_layer.count_errors(arr_result)
            else:
                arr_result += (self.grad * self.errors) ** 2.0


#....................##..........###....##....##.########.########...######....................#
#....................##.........##.##....##..##..##.......##.....##.##....##...................#
#....................##........##...##....####...##.......##.....##.##.........................#
#....................##.......##.....##....##....######...########...######....................#
#....................##.......#########....##....##.......##...##.........##...................#
#....................##.......##.....##....##....##.......##....##..##....##...................#
#....................########.##.....##....##....########.##.....##..######....................#


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
        Layer for adding two arrays
    '''

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
        self.arr_left.count_errors_if_grad_counted(arr_result)
        self.arr_left.zero_grad()
        self.arr_right.count_errors_if_grad_counted(arr_result)
        self.arr_right.zero_grad()


class SubLayer(Layer):
    '''
        Layer for subtracting two arrays
    '''

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
        self.arr_left.count_errors_if_grad_counted(arr_result)
        self.arr_left.zero_grad()
        self.arr_right.count_errors_if_grad_counted(arr_result)
        self.arr_right.zero_grad()


class MultiplyLayer(Layer):
    '''
        Layer for multiplying two arrays
    '''

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
        self.arr_left.count_errors_if_grad_counted(arr_result)
        self.arr_left.zero_grad()
        self.arr_right.count_errors_if_grad_counted(arr_result)
        self.arr_right.zero_grad()


class DegreeLayer(Layer):
    '''
        Layer for raising array to a degree
    '''

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
        push_grad = other_grad * self.deg * \
        self.arr.values ** (self.deg - 1)
        
        self.arr.backward(push_grad)

    def zero_grad_layer(self):
        self.arr.zero_grad()

    def count_errors(self, arr_result):
        self.arr.count_errors_if_grad_counted(arr_result)
        self.arr.zero_grad()


class DivideLayer(Layer):  # maybe it will be faster than x * y^{-1}
    '''
        Layer for dividing two arrays
    '''

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
        self.arr_left.count_errors_if_grad_counted(arr_result)
        self.arr_left.zero_grad()
        self.arr_right.count_errors_if_grad_counted(arr_result)
        self.arr_right.zero_grad()


class SelectLayer(Layer):
    '''
        Layer for getting element from an array
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, arr, key):
        self.arr = arr
        self.key = key
        result = array([self.arr.values[key]], [self.arr.errors[key]])
        result.grad_layer = self
        result.is_leaf = False

        return result

    def backward_layer(self, other_grad):
        push_grad = np.zeros_like(self.arr.values, dtype=np.float32)
        push_grad[self.key] = other_grad[0]
        self.arr.backward(push_grad)

    def zero_grad_layer(self):
        self.arr.zero_grad()

    def count_errors(self, arr_result):
        self.arr.count_errors_if_grad_counted(arr_result)
        self.zero_grad()


def exp(x):
    layer = ExpLayer()
    return layer(x)


def ExpLayer():
    '''
        Layer for raise exp to degree of array's elements
    '''

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
        self.arr.count_errors_if_grad_counted(arr_result)
        self.arr.zero_grad()


def log(x):
    layer = LogLayer()
    return layer(x)


class LogLayer():
    '''
        Layer for take logarithm of array's elements
    '''

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
        self.arr.count_errors_if_grad_counted(arr_result)
        self.arr.zero_grad()


def sqrt(x):
    layer = DegreeLayer()
    return layer(x, 0.5)
