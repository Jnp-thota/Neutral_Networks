
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
    df_Xtrain = df_Xtrain.sample(n=sample_size, random_state=42)
    y_train = y_train.sample(n=sample_size,random_state=42)
    y_test = y_test.sample(n=sample_size,random_state=42)
    df_Xtest = df_Xtest.sample(n=sample_size, random_state=42)

# # Randomly select indices for sampling
#     train_indices = np.random.choice(df_Xtrain.shape[0], sample_size, replace=False)
#     test_indices = np.random.choice(df_Xtest.shape[0], sample_size, replace=False)
    
#     print(train_indices.shape)

# # Select samples from X_train, y_train, X_test, and y_test arrays
#     df_Xtrain = df_Xtrain[train_indices]
#     y_train = y_train[train_indices]
#     df_Xtest = df_Xtest[test_indices]
#     y_test = y_test[test_indices]
    

    
    # sample_size = 1000  # Adjust the sample size based on your dataset size
  
    # y_train = y_train.sample(n=sample_size,random_state=42)
    # y_test = y_test.sample(n=sample_size,random_state=42)
    # df_Xtest = df_Xtest.sample(n=sample_size, random_state=42)
    
    def train_model(X_train, y_train, X_val, y_val, num_layers, num_nodes, epochs=100, learning_rate=0.01):
        model = Sequential()
        model.add(Linear(X_train.shape[1], num_nodes))
        model.add(ReLu())
    
        for _ in range(num_layers - 1):
            model.add(Linear(num_nodes, num_nodes))
            model.add(Sigmoid())
            
    
        model.add(Linear(num_nodes, 1))
        model.add(Sigmoid())
        loss_fn = BinaryCrossEntropyLoss()

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            print(y_train.shape, X_val.shape, y_val.shape)
            output_train = model.forward(X_train)
            print(output_train.shape)
            print(y_train.shape)
            loss_train = np.mean(loss_fn.forward(output_train, y_train))
            train_losses.append(loss_train)
            gradient = loss_fn.backward("")
            print(gradient.shape)
            model.backward(gradient)

            # Validation
            output_val = model.forward(X_val)
            loss_val = loss_fn.forward(output_val, y_val)
            val_losses.append(loss_val)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss {loss_train}, Val Loss {loss_val}")

        # Early stopping
            if len(val_losses) > 3 and val_losses[-1] > val_losses[-2] > val_losses[-3] > val_losses[-4]:
                print("Early stopping.")
                break

        return train_losses, val_losses
    
    # Experiment with different configurations
    configs = [(2, 32), (3, 64), (4, 128)]  # Example configurations (num_layers, num_nodes)
    all_train_losses = []
    all_val_losses = []
    
    X_train, X_val, y_train, y_val = train_test_split(df_Xtrain, y_train, test_size=0.2, random_state=42)

    
    
    for config in configs:
        print(f"Training model with {config[0]} layers and {config[1]} nodes per layer...")
        train_losses, val_losses = train_model(X_train, y_train, X_val, y_val, num_layers=config[0], num_nodes=config[1])
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
   