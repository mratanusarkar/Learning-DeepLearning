import random
from micrograd.engine import Value

# base module class
class Module:

    def zero_grad(self):
        """
        resets gradient to zero in the model parameter Value object
        """
        
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        return a list of model parameters
        """
        
        return []


# Karpathy's micrograd Neuron, Layer and MLP for Noobs!
class NoobNeuron(Module):

    def __init__(self, n_inputs, nonlin='', label='') -> None:
        """
        Karpathy's micrograd Neuron for Noobs!
        Constructs a neuron that has `n_inputs` number of inputs and assigns random weights & bias to the neuron

        Parameters
        ----------
            n_inputs : int
                number of inputs entering into the neuron
            
            nonlin : str
                non linearity or activation function of the neuron
                choices: 'tanh', 'relu', 'none'
                if left blank, tanh will be applied. if 'none', there won't be any nonlinearity.
            
            label : str
                (optional) the name or label of the Neuron
        """

        self.label = label
        self.nonlin = nonlin
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x) -> Value:
        """
        Takes in a list of all `n_inputs` number of input `x values`, 
        Computes the body by doing `sum(w.x + b)` and 
        Passes the result through a non-linear activation function to return output

        x = [x1, x2, x3, ..., xn],
        
        and we already have w's and b who's values were randomly assigned.
        w = [w1, w2, w3, ..., wn]
        b = bias

        so, the output would be as follows:
        out = f((x1*w1 + x2*w2 + ... + xn*wn) + b),
        or `out = f(Σ(w.x) + b)`

        where, f() is the activation function (tanh in this case)

        Parameters
        ----------
            x : list(float)
                the input values or the x = [x1, x2, x3, ..., xn]

            return : Value
                returns the output value
        """

        sum = Value(0.0, label=f'{self.label} : sum')
        for i, (xi, wi) in enumerate(zip(x, self.w)):
            # inputs
            x = xi if isinstance(xi, Value) else Value(xi)
            x.label=f'{self.label} : x{i}'

            # weights
            w = wi;         w.label=f'{self.label} : w{i}'

            # dendrites of the neuron (inputs weighted)
            den = x * w;    den.label=f'{self.label} : den{i}'

            # sum of dendrite signals inside cell body
            sum = sum + den;

        # bias of the neuron
        b = self.b;     b.label=f'{self.label} : b'

        # bias gets added inside the body
        body = sum + b;     body.label=f'{self.label} : body'

        # the net signal passes through a non-linear activation function
        if self.nonlin == 'tanh':
            out = body.tanh()
        elif self.nonlin == 'relu':
            out = body.relu()
        elif self.nonlin == 'none':
            out = body
        else:
            out = body.tanh() 
        out.label=f'{self.label} : out'

        return out
    
    def parameters(self):
        """
        return a list of params of the neuron: weights & biases
        """
        
        params = self.w + [self.b]
        return params

class NoobLayer(Module):

    def __init__(self, n_neurons_prev, n_neurons_curr, nonlin='', label='') -> None:
        """
        Karpathy's micrograd Layer for Noobs!
        Constructs a Layer, i.e array of Neurons, that has `n_neurons_curr` number of neurons in it.
        Each of the `n_neurons_curr` number of neurons in this layer has `n_neurons_prev` number of inputs and one output each.

        So, in total, we have `n_neurons_prev * n_neurons_curr` number of neural link lines feeding into the layer
        and `n_neurons_curr` number of links going out of the layer

        Parameters
        ----------
            n_neurons_prev : int
                number of neurons in the previous layer, or number of neurons entering into this current layer
            
            n_neurons_curr : int
                number of neurons in this current layer, or number of output lines exiting this layer
            
            nonlin : str
                non linearity or activation function of the neuron
                choices: 'tanh', 'relu', 'none'
                if left blank, tanh will be applied. if 'none', there won't be any nonlinearity.
            
            label : str
                (optional) the name or label of the Layer, 
                where each Neuron in the Layer will be labeled as: `{label} N:{i}`
        """
        
        self.label = label
        self.neurons = [NoobNeuron(n_neurons_prev, nonlin, f'{self.label} N:{i}') for i in range(n_neurons_curr)]

    def __call__(self, x) -> list[Value]:
        """
        Signals from the previous layer enters into this layer,
        and the output from all the previous layer's neurons are passed into each neuron of this current layer
        the output of each of the neurons in this current layer is computed
        and the result of the same is returned in an array

        If you have n neurons in this layer, your out will be an array of n
        x = [x1, x2, x3, ..., xn]

        Parameters
        ----------
            x : list(float)
                output of all `n_neurons_prev` neurons from the prev layer, that are entering into this layer.
            
            return : list(Value)
                returns a list of output values
        """

        outs = [n(x) for n in self.neurons]
        return outs
    
    def parameters(self):
        """
        return an extended list of all params in all the neurons of the layer
        """
        
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

class NoobMLP(Module):
    
    def __init__(self, n_inputs, neurons_per_layer, nonlin='', label='') -> None:
        """
        Karpathy's micrograd MLP for Noobs!
        Constructs a fully connected Multilayer Perceptron.

        Takes in number of inputs to the neural network `n_inputs`, and number of hidden & output layers `neurons_per_layer`,
        and creates a MLP with Layers and Neurons as per input specification, and assigns random weights and biases to the MLP.

        Parameters
        ----------
            n_inputs : int
                number of inputs to the entire MLP
            
            neurons_per_layer : list(int)
                a list specifying the number of neurons in each layer of the neural network
            
            nonlin : str
                non linearity or activation function of the neuron
                choices: 'tanh', 'relu', 'none'
                if left blank, tanh will be applied. if 'none', there won't be any nonlinearity.
            
            label : str
                (optional) the name or label of the entire MLP network (incase you want to create multiple networks and connect them) 
                each Neuron in the Layer will be labeled as: `{label} L:{i} N:{i} : <node-type>`,
                
                where: 
                - L is Layer
                - N is Neuron
                - <node-type> can be `w`, `b`, `body`, `sum`, `dendrite`, etc
                
        """

        self.label = label
        layer_sizes = [n_inputs] + neurons_per_layer
        self.layers = [NoobLayer(layer_sizes[i], layer_sizes[i+1], nonlin if i!=len(neurons_per_layer)-1 else 'none', f'{self.label} L:{i+1}') for i in range(len(neurons_per_layer))]

    def __call__(self, x) -> list[Value]:
        """
        An entire forward pass of the MLP,
        Where it creates and forms the entire neural network and the expression graph!

        pass in your input x that you want to feed to this MLP neural network, where
        x = [x1, x2, x3, ..., xn]

        Parameters
        ----------
            x : list(float)
                input features x = [x1, x2, x3, ..., xn]
            
            return : list(Value)
                an entire forward pass is made to the MLP with the given inputs, and the existing w&b of the network
                and it returns the output from the last output layer
        """

        # call the layers sequentially and make a forward pass
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """
        return an extended list of all params in all the layers of the mlp
        """
        
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params


# original micrograd Neuron, Layer and MLP that is similar to PyTorch API
class Neuron(Module):

    def __init__(self, nin, nonlin=True) -> None:
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        body = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = body.tanh() if self.nonlin else body
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, nin, nout, **kwargs) -> None:
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        outs = outs[0] if len(outs) == 1 else outs
        return outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):

    def __init__(self, nin, nouts) -> None:
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
