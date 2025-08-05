import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
for filename in os.listdir(path):
  if filename.endswith(".csv"):
    csv_file_path = os.path.join(path, filename)
    break
data = pd.read_csv(csv_file_path)
data.head()

#transforming the array of data into tensors
numeric_data = data.select_dtypes(include=[np.number])
tensor_data = tf.constant(numeric_data)
tensor_data = tf.cast(tensor_data, tf.float32)
print(tensor_data)

#setting up the x-values and the y-values
tensor_data = tf.random.shuffle(tensor_data)
x = tensor_data[:,:6]
y = tensor_data[:,-1]
y = tf.expand_dims(y, axis=1)
print(x.shape)
print(y.shape)

#creating ratios to split the datasets
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DATASET_SIZE = len(x)
print(DATASET_SIZE)

#THE TRAINING DATASETS
x_train = x[:int(TRAIN_RATIO * DATASET_SIZE)]
y_train = y[:int(TRAIN_RATIO * DATASET_SIZE)]
print(x_train.shape)
print(y_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

#THE VALIDATION DATASET
x_val = x[int(TRAIN_RATIO * DATASET_SIZE):int((TRAIN_RATIO+VAL_RATIO)*DATASET_SIZE)]
y_val = y[int(TRAIN_RATIO * DATASET_SIZE):int((TRAIN_RATIO + VAL_RATIO)* DATASET_SIZE)]
print(x_val.shape)
print(y_val.shape)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

#THE TESTING DATASET
x_test = x[int((TRAIN_RATIO + VAL_RATIO)*DATASET_SIZE):]
y_test = y[int((TRAIN_RATIO + VAL_RATIO)*DATASET_SIZE):]
print(x_test.shape)
print(y_test.shape)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

#normalizing the tenosr for later operations
normalizer = Normalization()
normalizer.adapt(x_train)
normalizer(x_train)

#THE PREDICTION MODEL
model = tf.keras.Sequential([
    InputLayer(shape=(6,)),
    normalizer,
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=1)
])
model.summary()

#optional arrangement to check the model
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes =True)

#compiling the model
model.compile(optimizer=Adam(learning_rate = 0.1), loss=MeanAbsoluteError(),metrics=[RootMeanSquaredError()])

#training the model
history = model.fit(train_dataset,validation_data=val_dataset, epochs=100,verbose=1)

#visualising the data gained after training the model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val_loss'])
plt.show()

#evaluate the models performance
model.evaluate(test_dataset)
model.predict(tf.expand_dims(x_test[1],axis=0))
print(y_test[1])
