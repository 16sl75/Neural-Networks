#implementing a simple perceptron from scratch

import numpy as np

#reading of data from files
def reading_data(string):
    with open(string, 'r') as f:
        data = f.readlines()

    all_data = []
    all_label = []

    for item in data:
        x = item.split(',')
        #obtaining label from data
        label = x[-1].rstrip()
        #obtaining features from data
        indv_data = x[:-1]
        indv_data = np.array(indv_data).astype(np.float)

        all_data.append(indv_data)
        
        label_vec = np.zeros(3)
        
        #converting labels into one-hot vectors
        if label == 'Iris-setosa':
            label_vec[0] = 1
            all_label.append(label_vec)
        elif label == 'Iris-versicolor':
            label_vec[1] = 1
            all_label.append(label_vec)
        elif label == 'Iris-virginica':
            label_vec[2] = 1
            all_label.append(label_vec)
    
    return (all_data, all_label)

#perceptron model
def model(data, label):
    #initial value of weights for each output node
    np.random.seed(0)
    w1 = np.random.rand(5)
    w2 = np.random.rand(5)
    w3 = np.random.rand(5)
    
    w1[0] *= 2
    
    confusion_matrix = np.zeros([3,3])
    
    #for pocket algorithm
    current_correct_1 = 0
    max_correct_1 = 0
    current_correct_2 = 0
    max_correct_2 = 0
    current_correct_3 = 0
    max_correct_3 = 0

    for i in range(len(data)):
        #split actual values of each output node
        d1 = label[i][0]
        d2 = label[i][1]
        d3 = label[i][2]
        
        y = np.zeros(3)
        
        #each data point is fed into the perceptrons
        #pocket algorithm without ratchet
        #output node 1
        f1, y[0] = predict(w1, data[i])
        if (y[0] == d1):
            current_correct_1 += 1
        elif (current_correct_1 > max_correct_1):
            max_correct_1 = current_correct_1
            best_w1 = w1
            current_correct_1 = 0
            w1 = adjust_weights(w1, data[i], y[0], d1)
    
        #output node 2
        f2, y[1] = predict(w2, data[i])
        if (y[1] == d2):
            current_correct_2 += 1
        elif (current_correct_2 > max_correct_2):
            max_correct_2 = current_correct_2
            best_w2 = w2
            current_correct_2 = 0
            w2 = adjust_weights(w2, data[i], y[1], d2)
            
        #output node 3
        f3, y[2] = predict(w3, data[i])
        if (y[2] == d3):
            current_correct_3 += 1
        elif (current_correct_3 > max_correct_3):
            max_correct_3 = current_correct_3
            best_w3 = w3
            current_correct_3 = 0
            w3 = adjust_weights(w3, data[i], y[2], d3)
        
        #if data point is classified as more than one class
        #take maximum value of f
        if (y[0] and y[1] and y[2]):
            max_index = np.argmax([f1, f2, f3])
            y = np.zeros(3)
            y[max_index] = 1
        elif (y[0] and y[1]):
            max_index = np.argmax([f1, f2])
            y = np.zeros(3)
            y[max_index] = 1
        elif (y[0] and y[2]):
            max_index = np.argmax([f1, f3])
            y = np.zeros(3)
            if (max_index == 1):
                max_index += 1    
            y[max_index] = 1
        elif (y[1] and y[2]):
            max_index = np.argmax([f2, f3])
            y = np.zeros(3)
            max_index += 1    
            y[max_index] = 1
            
        #calculation of confusion matrix for each class
        for j in range(len(label[i])):
            if (label[i][j] == 1):
                for k in range(len(y)):
                    if (y[k] == 1):
                        confusion_matrix[j][k] += 1
        
        #calculation of precision and recall for each class
        precision1 = confusion_matrix[0][0] / confusion_matrix.sum(axis = 0)[0]
        recall1 = confusion_matrix[0][0] / confusion_matrix.sum(axis = 1)[0]
        
        precision2 = confusion_matrix[1][1] / confusion_matrix.sum(axis = 0)[1]
        recall2 = confusion_matrix[1][1] / confusion_matrix.sum(axis = 1)[1]
        
        precision3 = confusion_matrix[2][2] / confusion_matrix.sum(axis = 0)[2]
        recall3 = confusion_matrix[2][2] / confusion_matrix.sum(axis = 1)[2]
        
    return (w1, w2, w3, precision1, recall1, precision2, recall2, precision3, recall3, confusion_matrix)
        
#activation function
def predict(w, x):
    f = w[0] + np.dot(w[1:], x)
    
    #threshold function
    if f >= 0:
        y = 1
    else:
        y = 0
    return f, y

#adjusting weights using error correction learning
def adjust_weights(w, x, y, d, lr = 0.01):
    v = np.insert(x, 0, 1)
    delta_w = (d - y) * lr * v
    w += delta_w
    return w

#running model on test data
def model_test(w1, w2, w3, data, label, output_file):
    confusion_matrix = np.zeros([3,3])
    
    for i in range(len(data)):
        #split actual values of each output node
        d1 = label[i][0]
        d2 = label[i][1]
        d3 = label[i][2]
        
        y = np.zeros(3)
        
        #each data point is fed into the perceptrons
        #output node 1
        f1, y[0] = predict(w1, data[i])
        
        #output node 2
        f2, y[1] = predict(w2, data[i])
            
        #output node 3
        f3, y[2] = predict(w3, data[i])
        
        #if data point is classified as more than one class
        if (y[0] and y[1] and y[2]):
            max_index = np.argmax([f1, f2, f3])
            y = np.zeros(3)
            y[max_index] = 1
        elif (y[0] and y[1]):
            max_index = np.argmax([f1, f2])
            y = np.zeros(3)
            y[max_index] = 1
        elif (y[0] and y[2]):
            max_index = np.argmax([f1, f3])
            y = np.zeros(3)
            if (max_index == 1):
                max_index += 1    
            y[max_index] = 1
        elif (y[1] and y[2]):
            max_index = np.argmax([f2, f3])
            y = np.zeros(3)
            max_index += 1    
            y[max_index] = 1
            
        #write label to output file
        output_write(output_file, data[i], y)
        
        #calculation of confusion matrix for each class
        for j in range(len(label[i])):
            if (label[i][j] == 1):
                for k in range(len(y)):
                    if (y[k] == 1):
                        confusion_matrix[j][k] += 1
        
        #calculation of precision and recall for each class
        precision1 = confusion_matrix[0][0] / confusion_matrix.sum(axis = 0)[0]
        recall1 = confusion_matrix[0][0] / confusion_matrix.sum(axis = 1)[0]
        
        precision2 = confusion_matrix[1][1] / confusion_matrix.sum(axis = 0)[1]
        recall2 = confusion_matrix[1][1] / confusion_matrix.sum(axis = 1)[1]
        
        precision3 = confusion_matrix[2][2] / confusion_matrix.sum(axis = 0)[2]
        recall3 = confusion_matrix[2][2] / confusion_matrix.sum(axis = 1)[2]
        
    return (precision1, recall1, precision2, recall2, precision3, recall3, confusion_matrix)

#writes predicted label to output file
def output_write(output_file, values, label_vec):
    max_index = np.argmax(label_vec)
    if max_index == 0:
        label = 'Iris-setosa'
    elif max_index == 1:
        label = 'Iris-versicolor'
    elif max_index == 2:
        label = 'Iris-virginica'
    
    for item in values:
        output_file.write(str(item) + ",")
    output_file.write(label + "\n")

train_data, train_label = reading_data('iris_train.txt')
w1, w2, w3, precision1_train, recall1_train, precision2_train, recall2_train, precision3_train, recall3_train, confusion_matrix_train = model(train_data, train_label)
print ("Final values of weights:", w1, w2, w3)
print ("Training confusion matrix:", confusion_matrix_train)
print ("Precision & recall for node 1:", precision1_train, recall1_train)
print ("Precision & recall for node 2:", precision2_train, recall2_train)
print ("Precision & recall for node 3:", precision3_train, recall3_train)

predicted_label = open('predicted_label.txt', "w")
test_data, test_label = reading_data('iris_test.txt')
precision1_test, recall1_test, precision2_test, recall2_test, precision3_test, recall3_test, confusion_matrix_test = model_test(w1, w2, w3, test_data, test_label, predicted_label)
print ("Testing confusion matrix:", confusion_matrix_test)
print ("Precision & recall for node 1:", precision1_test, recall1_test)
print ("Precision & recall for node 2:", precision2_test, recall2_test)
print ("Precision & recall for node 3:", precision3_test, recall3_test)
predicted_label.close()



