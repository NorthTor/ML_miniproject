import json
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import os

DATA_PATH = "../Features/mfcc_500_class_3.json"
save_or_load = 0    # 0 : the CNN trains and save the model
                    # 1 : the CNN load an existing model and predict

save_path = "ASC_CNN"
save_dir = os.path.dirname(save_path)

def import_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])

    return X, Y

def prepare_datasets(test_size, validation_size):
    #Import the data
    X, Y = import_data(DATA_PATH)
    # Create train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y)
    # Create train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size, stratify=Y_train)

    #Reshaping the sets in order to have a 3rd dimension, necessary for the CNN.
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_test, X_validation, Y_train, Y_test, Y_validation

def build_model(input_shape):

     #create model
     model = keras.Sequential()

     model.add(keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape))
     model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
     model.add(keras.layers.BatchNormalization())

     model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
     model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
     model.add(keras.layers.BatchNormalization())

     model.add(keras.layers.Conv2D(64, (2, 2), activation='relu', input_shape=input_shape))
     model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
     model.add(keras.layers.BatchNormalization())

     model.add(keras.layers.Flatten())
     model.add(keras.layers.Dense(64, activation='relu'))
     model.add(keras.layers.Dropout(0))

     model.add(keras.layers.Dense(3, activation='softmax'))

     return model

def predict(model, x, y):
    x = x[np.newaxis, ...]          # x -> (1,430, ..., 1). First dimension is the number of samples included in x.
    prediction = model.predict(x)   # In our case we only predict for one sample. We still have to add this dimension.

    #extract index with highest score
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index : {}, Predicted index : {}".format(y, predicted_index))
    return 0

if __name__=="__main__":
    #create train, test and validation sets
    X_train, X_test, X_validation, Y_train, Y_test, Y_validation = prepare_datasets(0.25, 0.2)

    #build CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    #print(input_shape)
    if save_or_load == 0:
        model = build_model(input_shape)

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=['accuracy'])
        model.fit(X_train, Y_train,
                  validation_data=(X_validation, Y_validation),
                  batch_size=32, epochs=30)
        model.save(save_path)
        test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
        print("Accuracy on test set is : {}".format(test_accuracy))

    elif save_or_load == 1:
        x = X_test[90]
        y = Y_test[90]
        model = keras.models.load_model(save_path)
        predict(model, x, y)