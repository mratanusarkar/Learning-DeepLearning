import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        self.label = label  # optional, Value variable/node name
        self.data = data
        self.grad = 0.0     # initially, the grad be zero (means no effect on output)

        # internal variables used for autograd graph construction
        self._prev = set(_children)     # previous node
        self._op = _op                  # the operation that produced this node
        self._backward = lambda: None   # to do the little piece of chain rule at each local node, backwards
                                        # by default _backward is None, as there is nothing to do at a leaf node

    def __repr__(self):
        return f"Value(label={self.label if self.label else 'Null'}, data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # f(x) = x + y, then f'(x) = 1
            # local derivative (= 1.0) * out's grad
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # f(x) = x * y, then f'(x) = y
            # local derivative (= switching the data of leaf nodes) * out's grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # f(x) = x^n, then f'(x) = n * x^(n-1)
            # local derivative (= nx^(n-1)) * out's grad
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        ex = math.exp(x)
        out = Value(ex, (self, ), 'exp')

        def _backward():
            # f(x) = exp(x), then f'(x) = exp(x)
            # local derivative of exp (exp(x)) * out's grad
            self.grad += ex * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # f(x) = ReLU(x) = max(0,x), then f'(x) = {x:1|x>0 and a:0|x<=0}
            # local derivative of ReLU (x > 0) * out's grad
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            # f(x) = tanh(x), then f'(x) = 1 - tanh(x)^2
            # local derivative of tanh (1 - tanh(x)**2) * out's grad
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def backward(self): # backpropagation
        # our topological sort function
        def topological_sort(node):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(node)
            return topo

        # base case
        self.grad = 1.0

        # and then traverse backwards, calling _backward() on the way
        for node in reversed(topological_sort(self)):
            node._backward()
