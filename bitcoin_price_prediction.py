# %%
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import numpy as np
import os, datetime, pickle

# %% 1. Data loading
DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

train_df = pd.read_csv(os.path.join(DATASET_PATH, 'gemini_BTCUSD_2020_1min_train.csv'))

# Method 1
test_df = pd.read_csv(os.path.join(DATASET_PATH, 'gemini_BTCUSD_2020_1min_test.csv'), sep=' ', header=None)
# Method 2
# test_df = pd.read_csv(os.path.join(DATASET_PATH, 'gemini_BTCUSD_2020_1min_test.csv'), delim_whitespace=True, header=None)
# Method 3
# test_df = pd.read_csv(os.path.join(DATASET_PATH, 'gemini_BTCUSD_2020_1min_test.csv'), delim=r'\s+', header=None)

test_df[1] = test_df[1] + ' ' + test_df[2]
test_df = test_df.drop(columns=2, axis=1)
test_df.columns = train_df.columns

# %% 2. Data inspection
# From inspecting the data, the data seems to be in reverse time order from 2021 until 2020
train_df.head(10)
train_df.tail(10)

train_df.info()
train_df.describe()
test_df.info()

train_df.isna().sum() # 41 NaN values in Open column
train_df.duplicated().sum() # No duplicated values
test_df.isna().sum() # No NaN values
test_df.duplicated().sum() # No duplicated values

# Visualize missing data
msno.bar(train_df)

# %% 3. Data cleaning
# Convert Open column to float
train_df['Open'] = pd.to_numeric(train_df['Open'], errors='coerce')
train_df.info()

# Visualize NaN values in Open column
train_df['Open'].isna().sum()
plt.figure()
plt.plot(train_df['Open'])
plt.show()

# Using interpolation to fill up NaN value
train_df['Open'] = train_df['Open'].interpolate(method='polynomial', order=2)
train_df.isna().sum()

# Reverse the dataset
train_df_rev = train_df[::-1].reset_index(drop=True)
test_df_rev = test_df[::-1].reset_index(drop=True)

## Method 2
# train_df_rev = train_df[::-1].values
# test_df_rev = test_df[::-1].values

# Visualize Open column from train dataset
plt.figure()
plt.plot(train_df_rev['Open'])
plt.title('Bitcoin Open Prices')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.show()

# Visualize Open column from test dataset
plt.figure()
plt.plot(test_df_rev['Open'])
plt.title('Bitcoin Open Prices')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.show()

# %% 4. Features selection
# Define Open column as the data that will be used
data = train_df_rev['Open']

# %% 5. Data pre-processing
# Expand the dimension of the data
data_exp = np.reshape(data.values, (-1, 1))

# Normalization
mm_scaler = MinMaxScaler()
data_exp = mm_scaler.fit_transform(data_exp)

# Cut the data into time frame
TIME_FRAME = 120
SEED = 12345
features = []
targets = []

for i in range(TIME_FRAME, len(data_exp)):
    features.append(data_exp[i-TIME_FRAME:i])
    targets.append(data_exp[i])

features = np.array(features)
targets = np.array(targets)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=SEED)

# %% Model development
model = Sequential()
model.add(LSTM(64, input_shape=X_train.shape[1:], return_sequences=True))
model.add(LSTM(64))
model.add(Dense(y_test.shape[1]))

# Model summary
model.summary()
plot_model(model, to_file='resources/model.png', show_shapes=True, show_layer_names=True)

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])

# Define callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=LOG_DIR)
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.001)
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# %%
# Model training
EPOCHS = 5
BATCH_SIZE = 64

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[tb, reduce_lr, es], epochs=EPOCHS, batch_size=BATCH_SIZE)

# %% Model evaluation
# Concatenate the Open column in the test and train data
concat_data = pd.concat((test_df_rev['Open'], train_df_rev['Open']))
test_data = concat_data[:len(test_df_rev)+TIME_FRAME]

# Normalize test data
test_data = np.reshape(test_data.values, (-1,1))
test_data = mm_scaler.transform(test_data)

# Cut test data
test_features = []
test_targets = []

for i in range(TIME_FRAME, len(test_data)):
    test_features.append(test_data[i-TIME_FRAME:i])
    test_targets.append(test_data[i])

test_features = np.array(test_features)
test_targets = np.array(test_targets)

# Model prediction
y_pred = model.predict(test_features)

# Inverse the normalization
y_pred = mm_scaler.inverse_transform(y_pred)
y_true = mm_scaler.inverse_transform(test_targets)

# Plot the prediction againts its label
plt.figure()
plt.plot(y_pred, color='b', label='Prediction')
plt.plot(y_true, color='r', label='True price')
plt.title('Bitcoin Prediction Againts True Values')
plt.ylabel('Bitcoin Prices')
plt.xlabel('Time')
plt.legend()
plt.show()

print('MAE:', mean_absolute_error(y_pred, y_true))
print('MAPE:', mean_absolute_percentage_error(y_pred, y_true))

# %% Model saving
# Save scaler
with open('mms.pkl', 'wb') as f:
    pickle.dump(mm_scaler, f)

# Save model
model.save('model.h5')
