import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


#print("Environment is set up successfully!")
r'''
#######LOAD AND PREPROCESS THE DATA##########
# Load the dataset
original_df = pd.read_csv(r"C:\Users\rykup\source\repos\NetworkSecurityProject\Data\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

#Clean column names
original_df.columns = original_df.columns.str.strip()

#Encode the 'Label' column
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
original_df['Label'] = label_encoder.fit_transform(original_df['Label'])
print("Label encoding complete. Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

#Fields to check
fields_to_check = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets', 
                   'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 
                   'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min', 
                   'Subflow Fwd Bytes', 'Subflow Bwd Bytes']

#Handle negative values
original_df[fields_to_check] = original_df[fields_to_check].clip(lower=0)

#Apply log transformation
import numpy as np
for field in fields_to_check:
    original_df[field] = np.log1p(original_df[field])  # log1p ensures non-negative log transformation

#Rescale to [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
original_df[fields_to_check] = scaler.fit_transform(original_df[fields_to_check])

#Save the preprocessed dataset
output_path = r"C:\Users\rykup\source\repos\NetworkSecurityProject\preprocessed_dataset_updated.csv"
original_df.to_csv(output_path, index=False)
print("Updated dataset saved successfully to:", output_path)


r'''


###########SPLIT THE DATASET INTO TRAINING AND TESTING SET##########
#Load the preprocessed dataset
file_path = r"C:\Users\rykup\source\repos\NetworkSecurityProject\preprocessed_dataset_updated.csv"
df_scaled = pd.read_csv(file_path)

#Split features (X) and labels (y)
X = df_scaled.drop(columns=['Label'])  # Features
y = df_scaled['Label']  # Labels

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#DEBUG: Verify the split
print("Dataset split successfully!")
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
print(f"Training labels size: {y_train.shape}")
print(f"Testing labels size: {y_test.shape}")

##### Debugging Steps Before Training #####
#Verify and clean X_train
print("Checking X_train for NaN or infinite values...")
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

#Check and convert y_train/y_test to numeric
print("Checking y_train for NaN or infinite values...")
print("y_train data type:", y_train.dtype)
y_train = pd.to_numeric(y_train, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = pd.to_numeric(y_test, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)

#Remove invalid indices in y_train
valid_indices = ~y_train.isnull()
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]

#DEBUG: Final check
print("Final check for NaN or infinite values:")
print("NaN in X_train:", X_train.isnull().sum().sum())
print("Inf in X_train:", np.isinf(X_train).sum().sum())


##### Debugging Steps Before Training #####
#DEBUG: Verify and clean X_train
print("Checking X_train for NaN or infinite values...")
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

#DEBUG: Verify and clean y_train
print("Checking y_train for NaN or infinite values...")
print("y_train data type:", y_train.dtype)
print("Sample values in y_train:", y_train.head())

valid_indices = ~y_train.isnull() & ~np.isinf(y_train)
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]

#DEBUG: Final checks
print("Final check for NaN or infinite values:")
print("NaN in X_train:", X_train.isnull().sum().sum())
print("Inf in X_train:", np.isinf(X_train).sum().sum())


##### DESIGN AND TRAIN NEURAL NETWORK #######

#Define the neural network architecture
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),  # First hidden layer
    Dense(64, activation='relu'),  # Second hidden layer
    Dense(1, activation='sigmoid')  # Output layer
])

#Ensure y_train and y_test are integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

#DEBUG: Verify class distribution
print("Class distribution in y_train:")
print(y_train.value_counts())

#Define class weights
class_weights = {
    0: len(y_train) / y_train.value_counts()[0],
    1: len(y_train) / y_train.value_counts()[1]
}
#DEBUG: Class weights
print("Class weights:", class_weights)


#Define the neural network architecture with batch normalization
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

#Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#Convert class weights to Python floats
class_weights = {
    0: float(len(y_train) / y_train.value_counts()[0]),
    1: float(len(y_train) / y_train.value_counts()[1])
}
print("Class weights:", class_weights)

#Ensure data is NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#DEBUG: inputs
print("X_train type:", type(X_train))
print("y_train type:", type(y_train))
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Unique values in y_train:", np.unique(y_train))
print("Class weights:", class_weights)

#Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    class_weight=class_weights
)

#Evaluate the model accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

#########PLOT THE TRAINING HISTORY OVER EPOCHS################
#Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("ddos_detection_model.h5")
print("Model saved successfully!")
