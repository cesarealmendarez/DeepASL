from l_convolution import Convolution
from l_maxpool import MaxPool
from l_softmax import Softmax
import pickle

nominal_filter_file_dir = "NominalFilters.txt"
nominal_weights_file_dir = "NominalWeights.txt"
nominal_biases_file_dir = "NominalBiases.txt"

nominal_filters_test = []
nominal_biases_test = []
nominal_weights_test = [] 

with open(nominal_filter_file_dir, 'rb') as f:
    nominal_filters_test = pickle.load(f)
    
with open(nominal_weights_file_dir, 'rb') as d:
    nominal_weights_test = pickle.load(d)
    
with open(nominal_biases_file_dir, 'rb') as e:
    nominal_biases_test = pickle.load(e) 

def trained_forward_propagation(image):
    convolution = Convolution(8, nominal_filters_test)
    maxpool = MaxPool()
    softmax = Softmax(13 * 13 * 8, 25, nominal_weights_test, nominal_biases_test)

    convolution_output = convolution.forward_propagation((image / 255) - 0.5)
    maxpool_output = maxpool.forward_propagation(convolution_output)
    softmax_output = softmax.forward_propagation(maxpool_output)

    return softmax_output    

