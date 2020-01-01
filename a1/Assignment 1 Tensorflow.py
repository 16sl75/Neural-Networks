#implementing a simple ANN model using Tensorflow

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

#reading of data from files
def reading_data(string):
    with open(string, 'r') as f:
        data = f.readlines()

    all_data = []
    all_label = []

    for item in data:
        x = item.split(',')
        label = x[-1].rstrip()
        indv_data = x[:-1]
        indv_data = np.array(indv_data).astype(np.float)

        all_data.append(indv_data)
        
        label_vec = np.zeros(3)
        
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

train_data, train_label = reading_data('iris_train.txt')
train_data = np.array(train_data)
train_label = np.array(train_label)

#building Keras model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3, input_dim = 4, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.fit(train_data, train_label, epochs = 20, batch_size = 1)

test_data, test_label = reading_data('iris_test.txt')
test_data = np.array(test_data)
test_label = np.array(test_label)

#using model to predict on testing data
predict_label = model.predict(test_data)
#calculation of confusion matrix
matrix = confusion_matrix(test_label.argmax(axis=1), predict_label.argmax(axis=1))
print("Testing confusion matrix:", matrix)

#calculation of precision and recall
precision1 = matrix[0][0] / matrix.sum(axis = 0)[0]
recall1 = matrix[0][0] / matrix.sum(axis = 1)[0]
        
precision2 = matrix[1][1] / matrix.sum(axis = 0)[1]
recall2 = matrix[1][1] / matrix.sum(axis = 1)[1]
        
precision3 = matrix[2][2] / matrix.sum(axis = 0)[2]
recall3 = matrix[2][2] / matrix.sum(axis = 1)[2]

print("Precision & recall for node 1:", precision1, recall1)
print("Precision & recall for node 2:", precision2, recall2)
print("Precision & recall for node 3:", precision3, recall3)


