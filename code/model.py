import numpy as np
import pandas as pd
import tensorflow as tf

import os

import matplotlib.pyplot as plt

#os.slabelstem('ls ../data')

def predict(model, test_input):
    prediction_p = model.predict(test_input)

    return (prediction_p > 0.5).astype('int')
    
def preprocess_data(df, normalize=True):
    #preprocessing

    cols = df.columns

    # convert "male" to 1 and "female" to 0
    df['label_num'] = (df['label'] == "male").astype('int')

    #exclude 'label' column
    att_cols = cols[:-1]

    mean = df[att_cols].mean()
    std = df[att_cols].std()  

    if normalize:
        df[att_cols]=(df[att_cols]-mean)/std      
    
    return df, att_cols, mean, std

def split_data(df, att_cols, train_ratio=0.7):

    # split data into training and testing set
    # should be splitted with balance between male and female samples!
    df_len = len(df) # 3168 = 1584 male and 1584 female
    
    #try train_test_split function from sklearn?

    training_input = df.loc[:int(df_len*train_ratio), att_cols]
    training_label = df.loc[:int(df_len*train_ratio), 'label_num']
    testing_input = df.loc[int(df_len*train_ratio)+1:, att_cols]
    testing_label = df.loc[int(df_len*train_ratio)+1:, 'label_num']
 
    return training_input, training_label, testing_input, testing_label

def model_fit(training_input, training_label, testing_input, testing_label, att_cols, plot_acc=True, verbose=1):
    
    hist = tf.keras.callbacks.History()
    es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience= 10)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_dim=len(att_cols)),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


    model.fit(training_input, training_label, epochs=100, batch_size=10, validation_data=(testing_input, testing_label), callbacks = [hist, es], verbose=verbose)

    if plot_acc:
        pd.DataFrame(hist.history).plot(secondary_y=['loss'])
        plt.show(block=False)

    test_loss = model.evaluate(testing_input, testing_label)
    print("\n%s: %.2f%%" % (model.metrics_names[1], test_loss[1]*100))
    return model

def save_model(model, checkpoint_path):
    # Save JSON config to disk
    # model architecture
    json_config = model.to_json()
    with open(os.path.join(checkpoint_path, 'model_config.json'), 'w') as json_file:
        json_file.write(json_config)

    # Save model weights to disk
    model.save_weights(os.path.join(checkpoint_path,'model_weights.h5'))
    print("Model saved in path: %s" % checkpoint_path)

def load_model(checkpoint_path):
    # Reload the model from the 2 files we saved
    with open(os.path.join(checkpoint_path, 'model_config.json')) as json_file:
        json_config = json_file.read()

    new_model = tf.keras.models.model_from_json(json_config)
    new_model.load_weights(os.path.join(checkpoint_path,'model_weights.h5'))

    print("Model loaded from path: %s" % checkpoint_path)
    return new_model

def train():
    df= pd.read_csv("../data/voice.csv")
    df, att_cols = preprocess_data(df)
    training_input, training_label, testing_input, testing_label = split_data(df, att_cols)

    print("Number of male training data: ", sum(training_label == 1))
    print("Number of female training data: ", sum(training_label == 0))

    model = model_fit(training_input, training_label, testing_input, testing_label, att_cols, plot_acc=True, verbose=2)

    checkpoint_path = "../saved_model"
    save_model(model, checkpoint_path)
