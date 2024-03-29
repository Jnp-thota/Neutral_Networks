
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
from ReLu import ReLu
from Linear import Linear
from Sequential import Sequential
from Sigmoid import Sigmoid
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import time

class preprocessing:
    def __init__(self,df_Xtrain,df_Xtest):
        self.X_train = df_Xtrain
        self.X_test = df_Xtest
        
    def dataformatechange(self):
        for df in [self.X_train,self.X_test]:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])  
    
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
        self.X_train['store_and_fwd_flag'] = self.X_train['store_and_fwd_flag'].replace({'N': 0, 'Y': 1})
        self.X_test['store_and_fwd_flag'] = self.X_test['store_and_fwd_flag'].replace({'N': 0, 'Y': 1})
        
        self.X_train.drop(columns=['id','dropoff_datetime','pickup_datetime'], inplace=True)
        self.X_test.drop(columns=['id','dropoff_datetime','pickup_datetime'], inplace=True)
            
    
    
    def addDistance(self,lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # Average radius of the earth in km
        
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

        dlat = lat2 - lat1
        dlng = lng2 - lng1

        # Haversine formula to calculate the distance
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        h = AVG_EARTH_RADIUS * c

        return h
    
    def format_data(self,X_train,y_train):
        input_features = X_train
        target_data = y_train

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(input_features)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(scaled_features, target_data, test_size=0.05, random_state=21)

        # Reshape target data
        y_train = y_train.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)

        return X_train, y_train, X_val, y_val
        
        
if __name__=='__main__':
    dataset = np.load("//Users//jayanagaprakashthota//Downloads//nyc_taxi_data.npy", allow_pickle=True).item()
    X_train, y_train,X_test , y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]
    df_Xtrain = X_train.dropna()
    df_Xtest = X_test.dropna()

    preprcs = preprocessing(df_Xtrain,df_Xtest)
    preprcs.dataformatechange()
    preprcs.categoricalEncode()
    
    df_Xtrain['distance'] = preprcs.addDistance(df_Xtrain['pickup_latitude'],df_Xtrain['pickup_longitude'],df_Xtrain['dropoff_latitude'],df_Xtrain['dropoff_longitude'])
    df_Xtest['distance'] = preprcs.addDistance(df_Xtest['pickup_latitude'],df_Xtest['pickup_longitude'],df_Xtest['dropoff_latitude'],df_Xtest['dropoff_longitude'])
    
    # Sample size
    sample_size = 1000
    y_train = y_train.dropna()
    y_test = y_test.dropna()
    
    # Standardization
    df_Xtrain = (df_Xtrain - df_Xtrain.mean()) / df_Xtrain.std()
    df_Xtest = (df_Xtest - df_Xtest.mean()) / df_Xtest.std()
    
    # Split data into training and validation sets
    X_train, y_train, X_val, y_val = preprcs.format_data(df_Xtrain,y_train)
    
    
    # Batch size
    batch_size = 32
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
    train_batches = create_batches(X_train, y_train, batch_size)

    
    model_configs = [
    {"name": "Model 1", "layers": [Linear(input_size=X_train.shape[1], output_size=128), ReLu(), Linear(input_size=128, output_size=1), Sigmoid()]},
    {"name": "Model 2", "layers": [Linear(input_size=X_train.shape[1], output_size=128), ReLu(), Linear(input_size=128, output_size=64), ReLu(), Linear(input_size=64, output_size=1), Sigmoid()]},
    {"name": "Model 3", "layers": [Linear(input_size=X_train.shape[1], output_size=64), ReLu(), Linear(input_size=64, output_size=32), ReLu(), Linear(input_size=32, output_size=1), Sigmoid()]}
]
    
    lst = []
    training_stats =[]
    model_info = {'val_loss': float('inf'), 'model': None}
   
    # Training loop
    for config in model_configs:
        for learning_rate in [0.001,0.01]:
            print("Training ", config["name"],end="")
            print(f"  with Learning rate : {learning_rate}")
            model = Sequential()
            for layer in config["layers"]:
                model.add(layer)
            epochs = 100
            train_losses = []
            val_losses = []
            for epoch in range(epochs):
                for X_batch, y_batch in train_batches:
                    Binary = BinaryCrossEntropyLoss()
                    y_pred = model.forward(X_batch)
                    loss = Binary.forward(y_pred, y_batch)
                    grad_loss = Binary.backward()
                    model.backward(grad_loss)
                    # Update weights and biases
                    for layer in model.layers:
                        if isinstance(layer, Linear):
                            layer.weights -= learning_rate * layer.grad_weights
                            layer.bias -= learning_rate * layer.grad_bias
                train_losses.append(loss)
                # Validation loss
                val_loss = 0
                y_pred = model.forward(X_val)
                val_loss = BinaryCrossEntropyLoss().forward(y_pred, y_val)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
                # Early stopping
                if epoch > 2 and val_losses[-1] >= val_losses[-2] >= val_losses[-3]:
                    print("Early stopping...")
                    break
            lst = [learning_rate, batch_size, train_losses[-1], val_losses[-1]]
            current_stat = [int(model_configs.index(config) + 1)] + lst
            training_stats.append(current_stat)
            print(training_stats)
            if val_loss < model_info['val_loss']:
                model_info['val_loss'] = val_loss
                model_info['model'] = model
    
    
    data= None
    if model_info['model'] is not None:          
        data =model_info['model']
    # Find the row with the minimum validation loss using min function with a lambda
    min_val_loss_row = min(training_stats, key=lambda row: row[-1])

    # Display the row with the minimum validation loss
    print(min_val_loss_row)
    
    model=Sequential()
    model=data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_Xtest)
    y_predict = model.forward(scaled_features)
    mse_score = mean_squared_error(y_test,y_predict)
    print("MSE Error :", mse_score)