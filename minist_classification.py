# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD  # add more optimizer if you need
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

from tensorflow.keras import initializers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(train_images.shape)  # number of images for traning set (60000, 28x28)
print(len(train_labels))  # number of images for traning set
print(train_labels)

print(test_images.shape)  # number of images for traning set (60000, 28x28)
print(len(test_labels))  # number of images for traning set
print(test_labels)

'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

# Hyperparameter
num_hidden_layer = 1  # ex) 2  # number of hidden layers
num_neurons_in_hidden_layer = [10]  # ex) [10,  10]
num_epochs = 3
# Select actications; https://www.tensorflow.org/api_docs/python/tf/keras/activations
activation_functions = ['tanh']  # ['relu', 'relu'] ['tanh', 'sigmoid'']
# Select optimizers; https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
learning_rate_alaph = 0.1  # often in the range between 0.0 and 1.0
# values between 0.5 and 0.98, depending on how smooth you want the convergence to the local optima (low values for noisy gradients, high values for smooth gradients).

momentum_rate = 1
opt = SGD(learning_rate=learning_rate_alaph, momentum=momentum_rate)
# ex) opt = Adam(learning_rate=learning_rate_alaph)

ls_enable = True  # False #True # False #True for L2 regularization
lambda_value = 0.01  # Between 0 and 0.1, such as 0.1, 0.001, 0.0001

dropout_enable = True  # False #True # False #True for dropout
prob_value = 0.20  # Between 0% and 30%

initializer = initializers.RandomNormal(mean=0., stddev=1.)
initializer_enable = False

##########################################################
############# Policy Line ### Don't change below codes ###
##########################################################

# Normalization (0 to 1)
train_images = train_images / 255.0
test_images = test_images / 255.0


def NN(train_images, train_labels, test_images,  test_labels):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    # hidden layer
    for i in range(num_hidden_layer):
        if ls_enable:
            if initializer_enable:
                model.add(Dense(num_neurons_in_hidden_layer[i],
                                activation=activation_functions[i], kernel_regularizer=l2(lambda_value), kernel_initializer=initializer))
            else:
                model.add(Dense(num_neurons_in_hidden_layer[i],
                                activation=activation_functions[i], kernel_regularizer=l2(lambda_value)))
        else:
            if initializer_enable:
                model.add(Dense(num_neurons_in_hidden_layer[i],
                                activation=activation_functions[i]), kernel_initializer=initializer)
            else:
                model.add(Dense(num_neurons_in_hidden_layer[i],
                                activation=activation_functions[i]))

    if dropout_enable:
        model.add(Dropout(prob_value))

    # hidden layer
    model.summary()

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    h = model.fit(train_images, train_labels, epochs=num_epochs)

    plt.plot(h.history['loss'])
    plt.ylabel('Accuracy evolution')
    plt.xlabel('Epochs')
    plt.show()

    return model


model = NN(train_images, train_labels, test_images, test_labels)

#############################################################

# Evaluation of model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('############################')
print('Test accuracy:', test_acc)
print('############################')

# Predictions
# due to multiclass classifiation
probability_model = Sequential([model, Softmax()])
predictions = probability_model.predict(test_images)

# print(predictions[0])
np.argmax(predictions[0])


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i],  test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

plt.show()
