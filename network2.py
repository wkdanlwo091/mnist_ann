import random
import numpy as np

class Network2(object):

    def __init__(self, sizes):##check
        self.num_layers = len(sizes)##check
        self.sizes = sizes## check
        self.w = [np.random.randn(y,x) for x, y  in zip(sizes[:-1], sizes[1:])]
        self.b = [np.random.randn(x,1) for x in sizes[1:]]

    def feed_forward(self, a): #### check
        for w,b in zip(self.w, self.b):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epoch, mini_batch_size, eta, test_data):
        for k in range(epoch):## check
            np.random.shuffle(training_data)  ## check
            mini_batches = [training_data[l : l+ mini_batch_size] for l in range(0, len(test_data), mini_batch_size)]

            for l in mini_batches:
                b_sum = [0]  ##check
                w_sum = [0]  ##check
                for w, b in zip(self.w, self.b):  ##check
                    w_sum.append(np.zeros_like(w))  ##check
                    b_sum.append(np.zeros_like(b))  ##check
                for i in l:
                    delta, a  = self.back_propagation(i)##check
                    for m in range(1,self.num_layers): ### iterator starts at 1
                        b_sum[m] = b_sum[m] +  delta[m]
                        w_sum[m] = w_sum[m] +  np.dot(delta[m], a[m-1].T)
                for i in range(1, self.num_layers):## check
                        self.w[i-1] = self.w[i-1] - (eta / mini_batch_size) * w_sum[i]  ## check
                        self.b[i-1] = self.b[i-1] - (eta / mini_batch_size) * b_sum[i]  ## check
            if test_data:
                print("epoch: {} =  {} out of {} =".format(k, self.evaluate(test_data), len(test_data)))

    def back_propagation(self, mini_batch):
        ###initialization
        first_input = mini_batch[0]
        answer = mini_batch[1]
        a = [first_input]
        z = [0]
        delta = [0]
        for i in self.b:
            a.append(np.zeros_like(i))
            z.append(np.zeros_like(i))
            delta.append(np.zeros_like(i))

        ##feed forward
        l = 1
        for w,b in zip(self.w, self.b):
            z[l] = (np.dot(w, a[l-1])+b)
            a[l] = sigmoid(z[l])
            l+=1

        ####Output error
        last_layer = self.num_layers-1
        delta[last_layer] = cost_function(a[last_layer],answer)* derivative_sigmoid(z[last_layer])##check

        ####back_propagation_error
        for i in range(1,last_layer): #### upstream * local stream
            delta[last_layer-i] = np.dot(self.w[last_layer-i].T, delta[last_layer-i+1]) * derivative_sigmoid(z[last_layer-i])
        return delta,a

    def evaluate(self, test_data):##check
        sum = 0
        for i in range(0,len(test_data)):
            if(np.argmax(self.feed_forward(test_data[i][0])) == np.argmax(test_data[i][1])):
                sum+=1
        return sum

def sigmoid(z): ## check
    return 1/(1.0+ np.exp(-z))

def derivative_sigmoid(z):## check
    return sigmoid(z)*(1 - sigmoid(z))

def cost_function(a, y):## check
    return a-y