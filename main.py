import tensorflow as tf
import keras

print("keras.__version__:",keras.__version__)
print("tf.__version__:",tf.__version__)

fashion_mnist=keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
print(x_train.shape)
print(x_train.dtype)
x_valid=x_train[:5000]/255.0
x_train_=x_train[5000:]/255.0
y_valid=y_train[:5000]
y_train_=y_train[5000:]
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
print(class_names[y_train[0]])
#_________________Model___________

# model= keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28,28]))
# model.add(keras.layers.Dense(300,activation=keras.activations.relu))
# model.add(keras.layers.Dense(100,keras.activations.relu))
# model.add(keras.layers.Dense(10,keras.activations.softmax))
model=keras.models.load_model("mnist_fashion_model.h5")

print(model.summary())

#________compile model_____________

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy])

##_______________train and validation_____________________
history_train=model.fit(x_train_,y_train_,epochs=30,validation_data=(x_valid,y_valid)) # validation_split=0.1
#class weight , sample_weight

#______plot train 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.DataFrame(history_train.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) #set the vertical range to [0,1]
# plt.show()
plt.savefig('train.png')
plt.close()

#_________evaluate model

model.evaluate(x_test,y_test)
x_new=x_test[:3]
y_proba=model.predict(x_new)
print(y_proba.round(2))

#___________________
# y_pred=model.predict_classes(x_new)
y_pred=model.predict(x_new) 
y_pred=np.argmax(y_pred,axis=1)
print(y_pred)
print(np.array(class_names)[y_pred])

# model.save("mnist_fashion_model.h5")



