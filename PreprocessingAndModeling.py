
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from ReLu import ReLu
from Linear import Linear
from Sequential import Sequential
from Sigmoid import Sigmoid

import matplotlib.pyplot as plt
import time

class preprocessing:
    def __init__(self,df_Xtrain,df_Xtest):
        self.X_train = df_Xtrain
        self.X_test = df_Xtest
        
    def dataformatechange(self):
        for df in [self.X_train,self.X_test]:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])  # Datetyping the date
    
        for df in [self.X_train,self.X_test]:
            # Extract month from the pickup datetime
            df['month'] = df['pickup_datetime'].dt.month
            # Extract ISO week number
            df['week'] = df['pickup_datetime'].dt.isocalendar().week
            # Extract day of the week
            df['weekday'] = df['pickup_datetime'].dt.weekday
            # Extract hour of the day
            df['hour'] = df['pickup_datetime'].dt.hour
            # Extract minute of the hour
            df['minute'] = df['pickup_datetime'].dt.minute
            # Calculate minute of the day
            df['minute_oftheday'] = df['hour'] * 60 + df['minute']
            # Drop the now redundant 'minute' column
            df.drop(['minute'], axis=1, inplace=True)
            
    def categoricalEncode(self):
        # List of categorical features to be one-hot encoded
        categorical_features = ['store_and_fwd_flag', 'vendor_id']

        # Loop through each categorical feature to apply one-hot encoding
        for feature in categorical_features:
            # Apply one-hot encoding to both train and test datasets
            self.X_train = pd.concat([self.X_train, pd.get_dummies(self.X_train[feature])], axis=1)
            self.X_test = pd.concat([self.X_test, pd.get_dummies(self.X_test[feature])], axis=1)
    
            # Remove the original categorical column after encoding
            self.X_train.drop([feature], axis=1, inplace=True)
            self.X_test.drop([feature], axis=1, inplace=True)
            
    
    
    def addDistance(self,lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # Average radius of the earth in km
    # Convert latitude and longitude from degrees to radians
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

        # Compute differences in coordinates
        dlat = lat2 - lat1
        dlng = lng2 - lng1

        # Haversine formula to calculate the distance
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        h = AVG_EARTH_RADIUS * c

        return h
        
        
if __name__=='__main__':
    dataset = np.load("//Users//jayanagaprakashthota//Downloads//nyc_taxi_data.npy", allow_pickle=True).item()
    X_train, y_train,X_test , y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]
    df_Xtrain = X_train
    df_Xtest = X_test

    preprcs = preprocessing(df_Xtrain,df_Xtest)
    preprcs.dataformatechange()
    preprcs.categoricalEncode()
    preprcs.addDistance
    df_Xtrain['distance'] = preprcs.addDistance(df_Xtrain['pickup_latitude'],df_Xtrain['pickup_longitude'],df_Xtrain['dropoff_latitude'],df_Xtrain['dropoff_longitude'])
    df_Xtest['distance'] = preprcs.addDistance(df_Xtest['pickup_latitude'],df_Xtest['pickup_longitude'],df_Xtest['dropoff_latitude'],df_Xtest['dropoff_longitude'])
    df_Xtrain['store_and_fwd_flag'] = df_Xtrain['store_and_fwd_flag'].replace({'N': 0, 'Y': 1})
    df_Xtest['store_and_fwd_flag'] = df_Xtest['store_and_fwd_flag'].replace({'N': 0, 'Y': 1})
    
    df_Xtrain.drop(columns=['id','dropoff_datetime','pickup_datetime'], inplace=True)
    df_Xtest.drop(columns=['id','dropoff_datetime','pickup_datetime'], inplace=True)
    
    # Sample size
    sample_size = 1000
    df_Xtrain = df_Xtrain
    y_train = y_train
    y_test = y_test
    df_Xtest = df_Xtest
    
    
    # Batch size
    batch_size = 256

# Split data into batches
    def create_batches(X, y, batch_size):
        num_batches = len(X) // batch_size
        batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batches.append((X[start:end], y[start:end]))
        if len(X) % batch_size != 0:
            batches.append((X[num_batches * batch_size:], y[num_batches * batch_size:]))
        return batches
    train_batches = create_batches(df_Xtrain, y_train, batch_size)
    test_batches = create_batches(df_Xtest, y_test, batch_size)
    
    model_configs = [
    {"name": "Model 1", "layers": [Linear(input_size=X_train.shape[1], output_size=64), ReLu(), Linear(input_size=64, output_size=1), Sigmoid()]},
    {"name": "Model 2", "layers": [Linear(input_size=X_train.shape[1], output_size=128), ReLu(), Linear(input_size=128, output_size=64), ReLu(), Linear(input_size=64, output_size=1), Sigmoid()]},
    {"name": "Model 3", "layers": [Linear(input_size=X_train.shape[1], output_size=32), ReLu(), Linear(input_size=32, output_size=32), ReLu(), Linear(input_size=32, output_size=1), Sigmoid()]}
]
    
    
    # Training loop
    for config in model_configs:
        print("Training", config["name"])
        model = Sequential()
        for layer in config["layers"]:
            model.add(layer)
        epochs = 100
        learning_rate = 0.001
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            total_train_loss = 0
            for X_batch, y_batch in train_batches:
                Binary = BinaryCrossEntropyLoss()
                y_pred = model.forward(X_batch)
                loss = Binary.forward(y_pred, np.array(y_batch).reshape(-1, 1))
                total_train_loss += loss
                grad_loss = Binary.backward()
                model.backward(grad_loss)
            # Update weights and biases
                for layer in model.layers:
                        if isinstance(layer, Linear):
                            layer.weights -= learning_rate * layer.grad_weights
                            layer.bias -= learning_rate * layer.grad_bias.reshape(layer.bias.shape)
            avg_train_loss = total_train_loss / len(train_batches)
            train_losses.append(avg_train_loss)
        # Validation loss
            total_val_loss = 0
            for X_batch, y_batch in test_batches:
                y_pred = model.forward(X_batch)
                loss = BinaryCrossEntropyLoss().forward(y_pred, np.array(y_batch).reshape(-1, 1))
                total_val_loss += loss
            avg_val_loss = total_val_loss / len(test_batches)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        # Early stopping
            if epoch > 2 and val_losses[-1] >= val_losses[-2] >= val_losses[-3]:
                print("Early stopping...")
                break