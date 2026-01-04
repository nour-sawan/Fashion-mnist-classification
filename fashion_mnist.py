##import the required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

##load the fashion mnist dataset from keras datasets
fashion_mnist=tf.keras.datasets.fashion_mnist

###load the training and testing data
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

##define the class names for the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

##Explore the data
print("Training images shape:", train_images.shape) #(60000,28,28)
print("Number of training images:", len(train_labels))##60000
print("Test images shape:", test_images.shape) ##(10000,28,28)
print("Number of test images:", len(test_labels))##10000

##preprocess the data 
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
##print(train_labels[0]) ##-->9(Ankle boot) Just for expirementation

##scale the images to a range of 0 to 1
train_images = train_images/255.0
test_images= test_images/255.0

##Visualize the first 25 images from the training set and display the class name below each image (for verification purposes)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i] ,  cmap=plt.cm.binary) #TODO: show the training image of index i 
    plt.xlabel(train_labels[i]) #TODO: show the training image class name
plt.show()

##Buildng the model 
model= tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])
##compile the model 
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
                            
##Train the model 
model.fit(train_images,train_labels,epochs=10)

##Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images,test_labels)
print('\nTest accuracy:', test_accuracy)

##Make predictions
predictions=model.predict(test_images)

print("Prediction for first test image:", np.argmax(predictions[0])) ##print the predicted class for the first test image
##argmax means ( find the index of the maximum value in the array , the position not the value )

##print("Actual label for first test image:",
print(test_labels[0])


##Define plot_image function
def plot_image(i, predictions_array, true_labels, images):
    true_label, img = true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)
    
    
##Define plot_value_array function
def plot_value_array(i, predictions_array, true_labels):
    true_label = true_labels[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

