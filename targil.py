from scipy.spatial import distance
import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import matplotlib
matplotlib.use('TkAgg')


def compute_euclidean_distance(image1, image2):
    return distance.euclidean(image1, image2)

#(a)
def predict(training_images, labels, query_image, k):
    images_distance = numpy.array([compute_euclidean_distance(
        training_images[i], query_image) for i in range(len(training_images))])
    lowest_k_indices = numpy.argpartition(images_distance, k)
    labels_countsort = numpy.array([0 for i in range(0, 10)])
    for i in range(0, k):
        label = labels[lowest_k_indices[i]]
        labels_countsort[int(label)] = labels_countsort[int(label)] + 1
    return labels_countsort.argmax()


mnist = fetch_openml('mnist_784')
data = numpy.array(mnist['data'])
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

#(b)
first_1k_train = numpy.array(train[:1000])
first_1k_labels = numpy.array(train_labels[:1000])
k = 10
error = 0
for index in range(0, len(test)):
    predict_label = predict(
        first_1k_train, first_1k_labels, test[index], k)
    if(predict_label != int(test_labels[index])):
        error = error + 1
positive_rate = 100 - error/len(test) * 100
print("Positive rate for k=0 is {}".format(positive_rate))

#(c)
first_1k_train = numpy.array(train[:1000])
first_1k_labels = numpy.array(train_labels[:1000])
k_range = range(1, 101)
positive_rate = [0 for k in range(len(k_range))]
for k in k_range:
    error = 0
    for index in range(0, len(test)):
        predict_label = predict(
            first_1k_train, first_1k_labels, test[index], k)
        if(predict_label != int(test_labels[index])):
            error = error + 1
    positive_rate[k-1] = 100 - error/len(test) * 100
plt.plot(k_range, positive_rate)
plt.ylabel("positive %")
plt.xlabel("k")
plt.show()

#(d)
k = 1
n_vector = range(100, 5100, 100)
positive_rate = [0 for k in range(len(n_vector))]
postive_rate_index = 0
for n in n_vector:
    train_n = numpy.array(train[:n])
    labels_n = numpy.array(train_labels[:n])
    error = 0
    for index in range(0, len(test)):
        predict_label = predict(
            train_n, labels_n, test[index], k)
        if(predict_label != int(test_labels[index])):
            error = error + 1
    positive_rate[postive_rate_index] = 100 - error/len(test) * 100
    postive_rate_index = postive_rate_index + 1
plt.plot(n_vector, positive_rate)
plt.ylabel("positive %")
plt.xlabel("number of training")
plt.show()

