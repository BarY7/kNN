from matplotlib import pyplot
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy


def compute_euclidean_comparator(image):  # image is 'raw' - no label attached.
    def compute_relative_euclid(image2):
        return compute_euclidean_distance(image, image2[0])
    return compute_relative_euclid


def compute_euclidean_distance(image1, image2):
    sum = 0
    for i in range(0, len(image1)):
        sum += (image1[i] - image2[i]) ** 2
    return sum ** 0.5


def predict(training_images, labels, query_image, k):
    comparator = compute_euclidean_comparator(query_image)
    images_with_labels = [(training_images[i], labels[i])
                          for i in range(0, len(training_images))]
    images_with_labels.sort(key=comparator)
    labels_countsort = numpy.array([0 for i in range(0, 10)])
    for i in range(0, k):
        labels_countsort[int(images_with_labels[i][1])
                         ] = labels_countsort[int(images_with_labels[i][1])] + 1
    return labels_countsort.argmax()


mnist = fetch_openml('mnist_784')
data = numpy.array(mnist['data'])
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

first_1k_train = numpy.array(train[:1000])
first_1k_labels = numpy.array(train_labels[:1000])
k_range = range(1, 101)
error_vector = [0 for k in range(len(k_range))]
for k in k_range:
    error = 0
    for index in range(0, len(test)):
        predict_label = predict(
            first_1k_train, first_1k_labels, test[index], k)
        if(predict_label != int(test_labels[index])):
            error = error + 1
    print("ERROR IS : {}".format(error/len(test) * 100))
    error[k-1] = error/len(test) * 100
print("####################ERROR VECTOR#######################", error_vector)
plt.plot(k_range, error_vector)
plt.ylabel("errors")
plt.xlabel("k")
plt.show()
print("here")
