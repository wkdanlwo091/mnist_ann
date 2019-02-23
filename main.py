from mnist_loader import *
import network
import network2

training_data, validation_data, test_data = load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

net = network2.Network2([784, 30,10]) ## 95 %
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
