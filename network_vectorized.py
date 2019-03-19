import random
import numpy as np
import json
import sys
import os

###dropout semi done?
###L2 regularization
###train minibatch set at once

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2
    @staticmethod
    def delta(z,a,y):
        return (a-y) * sigmoid_prime(z)
class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z,a, y):
        return (a-y)

class Network_vectorized(object):
    def __init__(self, sizes, cost = CrossEntropyCost):  ##check
        self.num_layers = len(sizes)  ##check
        self.sizes = sizes  ## check
        if (os.path.isfile('./parameter_save.json') == True):### w,b parameter exists?
            self.json_to_nparray()
        else: self.large_weight_initializer()
        self.dropout_position = [np.zeros_like(i) for i in self.b]
        self.cost = cost
    def default_weight_initilaizer(self):## slower one
        self.b = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.w = [np.random.randn(y,x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.v_b = [np.zeros((y,1)) for y in self.sizes[1:]]
        self.v_w = [np.zeros((y,x))/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    def large_weight_initializer(self):
        self.b = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.w = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.v_b = [np.zeros((y,1)) for y in self.sizes[1:]]
        self.v_w = [np.zeros((y,x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.nparray_to_json()
    def nparray_to_json(self):### save weights for future comparison
        json_b = [y.tolist() for y in self.b]
        json_w = [y.tolist() for y in self.w]
        json_v_b = [y.tolist() for y in self.v_b]
        json_v_w = [y.tolist() for y in self.v_w]
        a = [json_b, json_w, json_v_b, json_v_w]
        f = open("parameter_save.json", "w")
        json.dump(a, f)
        f.close()
    def json_to_nparray(self):
        print("opened good")
        f =  open("parameter_save.json", "r")
        tmp = json.loads(f.read())
        f.close()
        self.b = tmp[0]
        self.w = tmp[1]
        self.v_b = tmp[2]
        self.v_w  = tmp[3]
        self.b = [np.array(y) for y in self.b ]
        self.w = [np.array(y) for y in self.w]
        self.v_b = [np.array(y) for y in self.v_b]
        self.v_w = [np.array(y) for y in self.v_w]
    def feed_forward(self, a):  #### check
        for w, b in zip(self.w, self.b):
            a = sigmoid(np.dot(w, a) + b)
        return a
    def SGD(self, training_data, epoch, mini_batch_size, eta,\
            lmbda = 0.0,\
            evaluation_data=None,\
            monitor_evaluation_cost=False,\
            monitor_evaluation_accuracy=False,\
            monitor_training_cost=False,\
            monitor_training_accuracy=False, no_improvement_n = 10):

        list_eval = []
        n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy  = [], []
        training_cost, training_accuracy = [] , []
        max_accuracy = 0
        early_stopping = 0 ## for early stoppping

        for k in range(epoch):  ## check
            np.random.shuffle(training_data)  ## check
            mini_batches = [training_data[l: l + mini_batch_size] for l in range(0, len(training_data), mini_batch_size)]
            ####vectorized version
            i = 1
            for l in mini_batches:
                self.dropout()
                delta , a = self.back_propagation(l, mini_batch_size)
                for i in range(1, self.num_layers):
                    self.v_w[-i] = 0.7*self.v_w[-i] - eta/mini_batch_size*(delta[-i].dot(a[-i-1].T))### momentum update
                    self.v_b[-i] = 0.7*self.v_b[-i] - eta/mini_batch_size*(np.reshape( np.sum(delta[-i], axis = 1), (self.b[-i].size,-1)))##momentum update
                    self.w[-i] = (1 - eta * (lmbda / n)) * self.w[-i] + self.v_w[-i]###뒤에서부터 iteration 시작해서 -i 임.
                    self.b[-i] = self.b[-i] + self.v_b[-i]
                    ##self.w[-i] = (1-eta*(lmbda/n))*self.w[-i] - eta/mini_batch_size*(delta[-i].dot(a[-i-1].T))## weight decay and momentum update ##
                    ##self.b[-i] = self.b[-i] - eta/mini_batch_size*(np.reshape( np.sum(delta[-i], axis = 1), (self.b[-i].size,-1)))## problem b size
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training_data : {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert = True)
                training_accuracy.append(accuracy)
                print("Accuracy on training_data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert = True)
                evaluation_cost.append(cost)
                print("cost on evaluation data : {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data),n_data))
                if(max_accuracy < accuracy):
                    max_accuracy = accuracy
                    early_stopping = 0
                else:
                    early_stopping +=1
                if(early_stopping == no_improvement_n):
                    break
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    def dropout(self):
        self.dropout_b = [np.random.randint(2, size = i.shape) for i in self.b]
    def back_propagation(self, mini_batch, mini_batch_size):### minibatch array simultaneously
        ##feedforward
        ##input = [i[0] for i in mini_batch]
        ##answer = [i[1] for i in mini_batch]
        input = np.hstack([i[0] for i in mini_batch])
        answer = np.hstack([i[1] for i in mini_batch])
        z = []
        a = [input]
        delta = []
        i = 0
        for w, b in zip(self.w, self.b):
            mini_batch = w.dot(input)+b
            z1 = mini_batch
            input = sigmoid(z1)
            if (i == 0 or i < self.num_layers-2):
                input = input * self.dropout_b[i]###  multiply 0 or 1 by dropout
            delta1 = np.zeros_like(mini_batch)
            z.append(z1)
            a.append(input)
            delta.append(delta1)
            i+=1
        output = input
        ##output error
        delta[-1] = cross_entrophy(output,answer)/mini_batch_size * derivative_sigmoid(z[-1])##### divided by mini_batch_size
        ##backward propagation
        for i in range(2, self.num_layers):
            delta[-i] = np.dot(self.w[-i+1].T, delta[-i+1]) * derivative_sigmoid(z[-i])
            delta[-i] = delta[-i] * self.dropout_b[-i]##### multiply 0 or 1 by dropout
        return delta, a
    def evaluate(self, test_data):  ##check
        sum = 0
        for i in range(0, len(test_data)):
            if (np.argmax(self.feed_forward(test_data[i][0])) == np.argmax(test_data[i][1])):
                sum += 1
        return sum
    def accuracy(self, data, convert = False):
        if convert:
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    def total_cost(self, data, lmbda, convert = False):
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a,y )/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.w)
        return cost
    def save(self, filename):
        "Save the neural network to the file filename"
        data = {"sizes" : self.sizes,
                "weights": [w.tolist() for w in self.w],
                "biases": [b.tolist() for b in self.b],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
def derivative_sigmoid(z):  ## check
    return sigmoid(z) * (1 - sigmoid(z))
def cost_function(a, y):  ## check
    return a - y
def cross_entrophy(a, y):## not done
    return (a-y)/a*(1-a)
def soft_max(a, y):
    return 1
def load(filename):
    """Load a neural network from the file ''filename''. Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network_vectorized(data["sizes"], cost = cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
### Miscellaneous functions
def vectorized_result(j):
    """Return a 10 dimensional unit vector with a 1.0 in the jth postion
    and zeroes elswhere. This is used to convert a digit (0,....,9)
    into a corresponding desired output from the neural network"""
    e = np.zeros((10,1))
    e[j] = 1.0
    return e
def sigmoid(z):  ## check
    return 1 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))
def default(self, obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')
