#implementing a backpropagation algorithm in a multilayer perceptron

from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

#obtain flattened data for dense neural network
X, Y = fetch_openml('mnist_784', version = 1, return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 60000, test_size = 10000, shuffle = False)

#performing normalisation
X_train = X_train.astype('float32')
X_train/= 255
X_test = X_test.astype('float32')
X_test /= 255

#generation of one-hot labels for output
def one_hot(arr):
    all_labels = []
    for item in arr:
        label_arr = np.full((10), 0)
        label = int(item)
        label_arr[label] = 1
        all_labels.append(label_arr)
    all_labels = np.array(all_labels)
    return all_labels

#sigmoid activation function
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

#derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

sigmoid_vfunc = np.vectorize(sigmoid_derivative)

#converting labels to one-hot vectors
all_train_labels = one_hot(Y_train)
print(all_train_labels.shape)

#parameters
batch_size = 64
input_dim = 784
hidden_nodes_1 = 64
data_size = 60000
output_dim = 10
lr = 0.1
momentum = 0.1
epoch = 0
iteration = 3

X_train_edit = X_train[:data_size]
all_train_labels_edit = all_train_labels[:data_size]

#generate random weights & bias for hidden nodes
np.random.seed(2)
#range of hidden weights: [0, 0.01]
weights_hidden = np.random.rand(input_dim, hidden_nodes_1)/100
bias_hidden = np.random.rand(1, hidden_nodes_1)/100

#keeping track of previous weight changes for momentum
prev_weights_hidden = np.zeros((input_dim, hidden_nodes_1))

#generate random weights & bias for output nodes
np.random.seed(2)
#range of output weights: [0, 0.1]
weights_output = np.random.rand(hidden_nodes_1, output_dim)/10
bias_output = np.random.rand(1, output_dim)/10

#keeping track of previous weight changes for momentum
prev_weights_output = np.zeros((hidden_nodes_1, output_dim))

#confusion matrix
confusion_matrix_train = np.zeros((10,10))

for j in range(iteration):
    for i in range(0, data_size, batch_size):
        total_error = 0
        epoch += 1
        print(epoch)

        #split input into batches
        batch_input = X_train_edit[i:i+batch_size]
        batch_actual_output = all_train_labels_edit[i:i+batch_size]

        #calculation of activation at hidden node
        a_hidden = np.matmul(batch_input, weights_hidden) + bias_hidden

        #applying activation function to a hidden node
        y_hidden = np.zeros((a_hidden.shape[0], a_hidden.shape[1]))
        for i in range(a_hidden.shape[0]):
            for j in range(a_hidden.shape[1]):
                y_hidden[i][j] = sigmoid(a_hidden[i][j])

        #note that y_hidden is input to output layer
        #calculation of activation at output node
        a_output = np.matmul(y_hidden, weights_output) + bias_output

        #applying activation function to a output node
        y_output = np.zeros((a_output.shape[0], a_output.shape[1]))
        for i in range(a_output.shape[0]):
            for j in range(a_output.shape[1]):
                y_output[i][j] = sigmoid(a_output[i][j])

        #deriving predicted class using argmax
        for i in range(y_output.shape[0]):
            predict_class = np.argmax(y_output[i])
            actual_class= np.argmax(batch_actual_output[i])
            confusion_matrix_train[predict_class][actual_class] += 1

        #calculating (d-y) for output node
        error_arr = batch_actual_output - y_output
        for i in range(error_arr.shape[0]):
            for j in range(error_arr.shape[1]):
                #epsilon = 0.1
                if (abs(error_arr[i][j]) <= 0.1):
                    error_arr[i][j] = 0
                total_error += error_arr[i][j] **2

        #calculation of errors and deltas
        output_delta = np.multiply(error_arr, sigmoid_vfunc(a_output))
        hidden_error = np.matmul(output_delta, np.transpose(weights_output))
        hidden_delta = np.multiply(hidden_error, sigmoid_vfunc(a_hidden))

        #calculation of changes for weights and bias
        weights_output_change = lr * np.matmul(np.transpose(y_hidden), output_delta)
        weights_hidden_change = lr * np.matmul(np.transpose(batch_input), hidden_delta)

        bias_output_change = lr * np.sum(output_delta, axis = 0)
        bias_hidden_change = lr * np.sum(hidden_delta, axis = 0)

        #performing weights and bias changes
        weights_output += weights_output_change + momentum * prev_weights_output
        weights_hidden += weights_hidden_change + momentum * prev_weights_hidden
        bias_output += bias_output_change
        bias_hidden += bias_hidden_change
        prev_weights_output = weights_output_change
        prev_weights_hidden = weights_hidden_change

        print(total_error)
print(confusion_matrix_train)

#calculation of precision and recall
for i in range(10):
    precision = confusion_matrix_train[i][i]/confusion_matrix_train.sum(axis = 1)[i]
    recall = confusion_matrix_train[i][i]/confusion_matrix_train.sum(axis = 0)[i]
    print('Class %d   precision: %.4f    recall: %.4f' % (i, precision, recall))

#converting labels to one-hot vectors
all_test_labels = one_hot(Y_test)
print(all_test_labels.shape)

#testing of data
data_size = 10000

#confusion matrix
confusion_matrix_test = np.zeros((10,10))

for i in range(0, data_size, batch_size):
    total_error = 0
    epoch += 1
    print(epoch)
    
    #split input into batches
    batch_input = X_test[i:i+batch_size]
    batch_actual_output = all_test_labels[i:i+batch_size]
    
    #calculation of activation at hidden node
    a_hidden = np.matmul(batch_input, weights_hidden) + bias_hidden
    
    #applying activation function to a hidden node
    y_hidden = np.zeros((a_hidden.shape[0], a_hidden.shape[1]))
    for i in range(a_hidden.shape[0]):
        for j in range(a_hidden.shape[1]):
            y_hidden[i][j] = sigmoid(a_hidden[i][j])
    
    #print(y_hidden)
    
    #note that y_hidden is input to output layer
    #calculation of activation at output node
    a_output = np.matmul(y_hidden, weights_output) + bias_output
    
    #applying activation function to a output node
    y_output = np.zeros((a_output.shape[0], a_output.shape[1]))
    for i in range(a_output.shape[0]):
        for j in range(a_output.shape[1]):
            y_output[i][j] = sigmoid(a_output[i][j])
    
    #deriving predicted class using argmax
    for i in range(y_output.shape[0]):
        predict_class = np.argmax(y_output[i])
        actual_class= np.argmax(batch_actual_output[i])
        confusion_matrix_test[predict_class][actual_class] += 1

print(confusion_matrix_test)

#calculation of precision and recall
for i in range(10):
    precision = confusion_matrix_test[i][i]/confusion_matrix_test.sum(axis = 1)[i]
    recall = confusion_matrix_test[i][i]/confusion_matrix_test.sum(axis = 0)[i]
    print('Class %d   precision: %.4f    recall: %.4f' % (i, precision, recall))



