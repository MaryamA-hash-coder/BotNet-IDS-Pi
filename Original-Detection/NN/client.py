import pandas as pd
import pickle
import socket
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split  
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

import psutil
import GPUtil
import time
import csv
import os
import threading



class ResourceMonitor:
    def __init__(self, log_file):
        self.log_file = log_file
        self.monitoring = False

    def start(self):
        self.monitoring = True
        self.start_time = time.time()
        with open(self.log_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Elapsed_Time", "CPU_Usage", "RAM_Usage", "GPU_Usage", "Disk_Read", "Disk_Write",
                             "Net_Sent", "Net_Recv", "CPU_Temp", "GPU_Temp"])

        threading.Thread(target=self.monitor_resources).start()

    def stop(self):
        self.monitoring = False

    def monitor_resources(self):
        while self.monitoring:
            elapsed_time = time.time() - self.start_time
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

            with open(self.log_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([elapsed_time, cpu_usage, ram_usage, gpu_usage, disk_read, disk_write,
                                 net_sent, net_recv, cpu_temp, gpu_temp])

            time.sleep(1)



def plot_resource_usage(log_file, phase_name):
    data = pd.read_csv(log_file)

    plt.figure()
    plt.plot(data["Elapsed_Time"], data["CPU_Usage"], label="CPU Usage (%)")
    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("CPU Usage (%)")
    plt.title(f"CPU Usage Over Time ({phase_name})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cpu_usage_{phase_name}.png")
    plt.close()

    plt.figure()
    plt.plot(data["Elapsed_Time"], data["RAM_Usage"], label="RAM Usage (%)", color="orange")
    plt.xlabel("Elapsed Time (seconds)")
    plt.ylabel("RAM Usage (%)")
    plt.title(f"RAM Usage Over Time ({phase_name})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ram_usage_{phase_name}.png")
    plt.close()

    if "GPU_Usage" in data.columns and data["GPU_Usage"].notna().any():
        plt.figure()
        plt.plot(data["Elapsed_Time"], data["GPU_Usage"], label="GPU Usage (%)", color="green")
        plt.xlabel("Elapsed Time (seconds)")
        plt.ylabel("GPU Usage (%)")
        plt.title(f"GPU Usage Over Time ({phase_name})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gpu_usage_{phase_name}.png")
        plt.close()

    print(f"Saved resource usage plots for {phase_name}.")



def train_neural_network(train_data_path, batch_size=32, epochs=1):
    train_data = pd.read_csv(train_data_path)

    X_train = train_data.drop(columns='label').values
    y_train = train_data['label'].values


    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)



    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(8, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    return model



def build_neural_network(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_validation(train_data_path, n_splits=3, batch_size=32, epochs=1):
    train_data = pd.read_csv(train_data_path)

    X = train_data.drop(columns='label').values
    y = train_data['label'].values

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_neural_network(input_dim=X.shape[1])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

        y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

        print(f"Fold {fold} Accuracy: {acc:.6f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    std_error = std_acc / np.sqrt(n_splits)

    print(f"\nMean Accuracy: {mean_acc:.6f}")
    print(f"Standard Deviation: {std_acc:.6f}")
    print(f"Standard Error: {std_error:.6f}")

    return mean_acc, std_acc, std_error



def test_neural_network(model, test_data_path):
    test_data = pd.read_csv(test_data_path)
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)

    X_test = test_data.drop(columns='label').values
    y_test = test_data['label'].values

    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    y_pred_prob = model.predict(X_test).flatten()

    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, cm, y_test, y_pred, y_pred_prob



def Performance(X_test, y_test_d, y_pred_d):
    y_test = y_test_d

    y_pred = (y_pred_d >= 0.5).astype(int)  

    print('Accuracy: %.6f' % metrics.accuracy_score(y_test, y_pred))
    print('Precision: %.6f' % metrics.precision_score(y_test, y_pred, average='weighted'))
    print('Recall: %.6f' % metrics.recall_score(y_test, y_pred, average='weighted'))
    print('F1 Score: %.6f' % metrics.f1_score(y_test, y_pred, average='weighted'))
    print('Hamming Loss: %.6f' % metrics.hamming_loss(y_test, y_pred))
    print('Jaccard Score: %.6f' % metrics.jaccard_score(y_test, y_pred, average='weighted'))


    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_d)
    auc_score = metrics.roc_auc_score(y_test, y_pred_d)
    print('AUC Score: %.6f' % auc_score)


    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %.6f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    print("ROC Curve saved as 'roc_curve.png' in the current directory.")


    cm = confusion_matrix(y_test, (y_pred_d >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion Matrix saved as 'confusion_matrix.png' in the current directory.")


    unique_classes = np.unique(y_test)
    print(f"Actual classes in y_test: {unique_classes}")
    classification_rep = metrics.classification_report(y_test, y_pred, labels=unique_classes, target_names=[str(c) for c in unique_classes], zero_division=1, digits=5)
    print(classification_rep)
    

# Paths
data_path = '~/Documents/Thesis/Original-Detection/TRAINING_SAMPLE_ORIGINAL4_MERGED.csv'
test_data_path = '~/Documents/Thesis/Original-Detection/TEST_SAMPLE_ORIGINAL4.csv'

# Train the model
if __name__ == "__main__":

    #mean_acc, std_dev, std_error = train_validation(data_path, epochs=5)

    train_monitor = ResourceMonitor("train_resource_usage.csv")
    train_monitor.start()

    nn_model = train_neural_network(data_path)
    train_monitor.stop()

    plot_resource_usage("train_resource_usage.csv", "Training")

    test_monitor = ResourceMonitor("test_resource_usage.csv")
    test_monitor.start()

    test_accuracy, test_precision, test_recall, test_cm, y_test, y_pred, y_pred_prob = test_neural_network(nn_model, test_data_path)

    test_monitor.stop()

    print(f"Neural Network Model Accuracy on Test Data: {test_accuracy}")
    print(f"Precision on Test Data: {test_precision}")
    print(f"Recall on Test Data: {test_recall}")
    print("Confusion Matrix on Test Data:")
    print(test_cm)

    plot_resource_usage("test_resource_usage.csv", "Testing")

    test_data = pd.read_csv(test_data_path)
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)
    X_test = test_data.drop(columns='label')
    y_test = test_data['label']
    
    Performance(X_test, y_test, y_pred)
