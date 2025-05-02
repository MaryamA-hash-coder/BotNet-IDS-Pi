import pandas as pd
import pickle
import socket
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split  
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold

import psutil
import GPUtil
import time
import csv
import os
import matplotlib.pyplot as plt
import threading


def monitor_resources(log_file="client_resource_usage.csv", interval=1):
    start_time = time.time()
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Elapsed_Time", "CPU_Usage", "RAM_Usage", "GPU_Usage", "Disk_Read", "Disk_Write",
                         "Net_Sent", "Net_Recv", "CPU_Temp", "GPU_Temp"])

        while True:
            elapsed_time = time.time() - start_time
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().percent
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else None
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes
            disk_write = disk_io.write_bytes
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv

            cpu_temp = None
            gpu_temp = None
            try:
                temp_file = "/sys/class/thermal/thermal_zone0/temp"
                if os.path.exists(temp_file):
                    with open(temp_file, "r") as f:
                        cpu_temp = float(f.read()) / 1000.0
                if gpus:
                    gpu_temp = gpus[0].temperature
            except Exception as e:
                print(f"Error reading temperature: {e}")

            writer.writerow([elapsed_time, cpu_usage, ram_usage, gpu_usage, disk_read, disk_write,
                             net_sent, net_recv, cpu_temp, gpu_temp])
            file.flush()
            time.sleep(interval)

def plot_resource_usage(log_file="client_resource_usage.csv"):
    data = pd.read_csv(log_file)

    plt.figure()
    plt.plot(data["Elapsed_Time"], data["CPU_Usage"], label="CPU Usage (%)")
    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cpu_usage.png")
    plt.close()
    print("Saved CPU Usage plot as 'cpu_usage.png'")

    plt.figure()
    plt.plot(data["Elapsed_Time"], data["RAM_Usage"], label="RAM Usage (%)", color="orange")
    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("RAM Usage (%)")
    plt.title("RAM Usage Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ram_usage.png")
    plt.close()
    print("Saved RAM Usage plot as 'ram_usage.png'")

    if "GPU_Usage" in data.columns and data["GPU_Usage"].notna().any():
        plt.figure()
        plt.plot(data["Elapsed_Time"], data["GPU_Usage"], label="GPU Usage (%)", color="green")
        plt.xlabel("Elapsed Time (seconds)")
        plt.ylabel("GPU Usage (%)")
        plt.title("GPU Usage Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("gpu_usage.png")
        plt.close()
        print("Saved GPU Usage plot as 'gpu_usage.png'")
    else:
        print("No GPU Usage data found in the log.")

    plt.figure()
    plt.plot(data["Elapsed_Time"], data["Disk_Read"], label="Disk Read (bytes)", color="blue")
    plt.plot(data["Elapsed_Time"], data["Disk_Write"], label="Disk Write (bytes)", color="red")
    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("Disk I/O (bytes)")
    plt.title("Disk I/O Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("disk_io.png")
    plt.close()
    print("Saved Disk I/O plot as 'disk_io.png'")

    plt.figure()
    plt.plot(data["Elapsed_Time"], data["Net_Sent"], label="Network Sent (bytes)", color="purple")
    plt.plot(data["Elapsed_Time"], data["Net_Recv"], label="Network Received (bytes)", color="brown")
    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("Network I/O (bytes)")
    plt.title("Network I/O Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("network_io.png")
    plt.close()
    print("Saved Network I/O plot as 'network_io.png'")

    if "CPU_Temp" in data.columns and data["CPU_Temp"].notna().any():
        plt.figure()
        plt.plot(data["Elapsed_Time"], data["CPU_Temp"], label="CPU Temperature (°C)", color="magenta")
        if "GPU_Temp" in data.columns and data["GPU_Temp"].notna().any():
            plt.plot(data["Elapsed_Time"], data["GPU_Temp"], label="GPU Temperature (°C)", color="cyan")
        plt.xlabel("Elapsed Time (seconds)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("temperature.png")
        plt.close()
        print("Saved Temperature plot as 'temperature.png'")
    else:
        print("No Temperature data found in the log.")

# Train a local neural network
def train_local_model(data, epochs=1, batch_size=32):
    X = data.drop(columns='Attack')
    y = data['Attack']

    y = to_categorical(y, num_classes=3)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(5,)),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
#        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        #tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dense(3, activation=tf.nn.softmax),
    ])

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=epochs)

    return model.get_weights(), X_val, y_val



def train_validation(train_data_path, n_splits=3, epochs=1, batch_size=32):
    data = pd.read_csv(train_data_path)
    
    X = data.drop(columns='Attack').values
    y = data['Attack'].values
    num_classes = 3
    y_encoded = to_categorical(y, num_classes=num_classes)

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y[val_idx]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(8, activation=tf.nn.relu),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax),
        ])

        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['categorical_accuracy'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred_prob = model.predict(X_val)
        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = accuracy_score(y[val_idx], y_pred)
        accuracies.append(acc)

        print(f"Fold {fold} Accuracy: {acc:.6f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    std_error = std_acc / np.sqrt(n_splits)

    print(f"\nMean Accuracy: {mean_acc:.6f}")
    print(f"Standard Deviation: {std_acc:.6f}")
    print(f"Standard Error: {std_error:.6f}")

    return mean_acc, std_acc, std_error




def send_weights_to_aggregator(data_path, server_ip, port=5000, epochs=1):
    client_data = pd.read_csv(data_path)

    client_weights, _, _ = train_local_model(client_data, epochs=epochs)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_ip, port))
        s.sendall(pickle.dumps(client_weights))
    print("Client weights sent to aggregator.")


# Paths and server details
client_id = 3  # Update with appropriate client ID (1, 2, 3, ...)
data_path = f'~/Documents/Thesis/FL-DT-Classification/Client{client_id}/TRAINING_SAMPLE_ORIGINAL4_Group{client_id}.csv'
server_ip = "192.168.1.124"

# Train the model and send to the aggregator
if __name__ == "__main__":

    #mean_acc, std_dev, std_error = train_validation(data_path, epochs=1, batch_size=32)


    monitoring_thread = threading.Thread(target=monitor_resources, args=(f"client{client_id}_resource_usage.csv",), daemon=True)
    monitoring_thread.start()

    # Load client-specific data
    #client_data = pd.read_csv(data_path)
    #epochs = 15

    # Train the local model
    #local_model, accuracy, _, _, _ = train_neural_network(client_data, epochs=epochs)
    #print(f"Client {client_id} - Model Accuracy: {accuracy}")

    send_weights_to_aggregator(data_path, server_ip, epochs=15)

    plot_resource_usage(f"client{client_id}_resource_usage.csv")
