#implements the LVQ algorithm

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
        
        #converting labels into one-hot vectors
        if label == 'Iris-setosa':
            label_vec = 0
            all_label.append(label_vec)
        elif label == 'Iris-versicolor':
            label_vec = 1
            all_label.append(label_vec)
        elif label == 'Iris-virginica':
            label_vec = 2
            all_label.append(label_vec)
    
    return (all_data, all_label)

#calculating euclidean distance
def euclidean_distance(input_vec, weights):
    distance = 0
    for i in range(len(weights)):
        distance += (input_vec[i] - weights[i])**2
    return np.sqrt(distance)

#shuffling of data and labels
def shuffle(data, labels):
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = np.array(data)
    labels = np.array(labels)
    return data[index], labels[index]

f = open("LVQ1.txt", "w")

all_data, all_label = reading_data('iris_train.txt')
all_data, all_label = shuffle(all_data, all_label)

test_data, test_label = reading_data('iris_test.txt')
test_data, test_label = shuffle(test_data, test_label)

#initialising weights and other parameters
np.random.seed(4)
weights = np.random.rand(3,4)
print(weights)
alpha = 0.3
n_epochs = 50

for epoch in range(n_epochs):
    cluster_error = np.zeros(3)
    lr = alpha * (1 - epoch/n_epochs)
    
    for i in range(len(all_data)):
        #calculating euclidean distance between data and each node
        distances = []
        for j in range(3):
            distances.append(euclidean_distance(all_data[i], weights[j]))
        #obtaining the predicted label
        predicted_label = int(distances.index(min(distances)))
        actual_label = int(all_label[i])

        #calculation of mean squared error
        error = all_data[i] - weights[predicted_label]
        mean_squared_error = np.dot(error, error)
        cluster_error[predicted_label] += mean_squared_error
        
        #performing weight change
        if (predicted_label == actual_label):
            delta_w = lr * error
        else:
            delta_w = -lr * error
        weights[predicted_label] += delta_w
        
        #weights normalisation
        weights[predicted_label] /= np.linalg.norm(weights[predicted_label])

    print("epoch:", epoch, "error:", cluster_error)
    f.write("epoch: " + str(epoch) + " error: " + str(cluster_error) + "\n")

print("weights", weights)
f.write("codebook vector 0: " + str(weights[0]) + "\n")
f.write("codebook vector 1: " + str(weights[1]) + "\n")
f.write("codebook vector 2: " + str(weights[2]) + "\n")

#calculation of confusion matrix
confusion_matrix = np.zeros([3,3])
correct_predictions = 0

#testing on test data
for i in range(len(test_data)):
    distances = []
    for j in range(3):
        distances.append(euclidean_distance(test_data[i], weights[j]))
    predicted_label = int(distances.index(min(distances)))
    actual_label = int(test_label[i])
    if (int(predicted_label) == int(actual_label)):
        correct_predictions += 1
    
    confusion_matrix[predicted_label][actual_label] += 1

print(confusion_matrix)
print("accuracy", correct_predictions/len(test_data))

f.close()

