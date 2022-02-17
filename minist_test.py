# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD  # add more optimizer if you need
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# pip3 install keras-tuner
import kerastuner as kt


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
num_hidden_layer = 2  # number of hidden layers
num_neurons_in_hidden_layer = [128,  10]
num_epochs = 5
# Select actications; https://www.tensorflow.org/api_docs/python/tf/keras/activations
activation_functions = ['relu', None]  # ['relu', 'tanh'] ['relu', 'sigmoid'']
# Select optimizers; https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
learning_rate_alaph = 0.001
opt = Adam(learning_rate=learning_rate_alaph)
# ex) opt = SGD(learning_rate=learning_rate_alaph)


##########################################################
############# Policy Line ### Don't change below codes ###
##########################################################

# Normalization (0 to 1)
train_images = train_images / 255.0
test_images = test_images / 255.0


def model_builder(hp):
    model = Sequential()

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units_1 = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_units_2 = hp.Int('units', min_value=32, max_value=512, step=32)

    model.add(Flatten(input_shape=(28, 28))) ## input layer
    model.add(Dense(units=hp_units_1, activation='relu'))
    model.add(Dense(units=hp_units_2, activation='relu'))

    model.add(Dense(10))  # output labels
    model.summary()

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='keras_tuner_dir',
                     project_name='wonse_zzang')

tuner.search(train_images, train_labels, epochs=10, validation_data=(
    test_images, test_labels))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

#### Display results
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels))




#############################################################

# Evaluation of model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('############################')
print('Test accuracy:', test_acc)
print('############################')

# Predictions
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

