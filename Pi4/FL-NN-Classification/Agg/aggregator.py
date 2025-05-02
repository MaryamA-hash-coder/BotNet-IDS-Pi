import pickle
import socket
import threading
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix,precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import psutil
import GPUtil
import time
import csv
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize



def monitor_resources(log_file="aggregator_resource_usage.csv", interval=1):
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


def plot_resource_usage(log_file="aggregator_resource_usage.csv"):
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



def federated_averaging(client_weights):

    averaged_weights = []
    for weights in zip(*client_weights):
        averaged_weights.append(np.mean(weights, axis=0))
    return averaged_weights




def receive_client_weights(port=5000, num_clients=3):
    client_weights = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('', port))
        server_socket.listen(num_clients)
        print(f"Aggregator listening on port {port}...")
        while len(client_weights) < num_clients:
            conn, addr = server_socket.accept()
            with conn:
                print(f"Received connection from {addr}")
                weights_data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    weights_data += chunk
                client_weights.append(pickle.loads(weights_data))
                print(f"Weights from {addr} received")
    return client_weights


def Performance(X_test, y_test_d, y_pred_d, optimize_threshold=True):
    def find_best_thresholds(y_test_bin, y_pred_bin):
        """
        Find the best threshold for each class based on F1 score.
        """
        best_thresholds = []
        for i in range(y_pred_bin.shape[1]):
            precision, recall, thresholds = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  
            best_idx = np.argmax(f1_scores)
            best_thresholds.append(thresholds[best_idx] if best_idx < len(thresholds) else 0.5)
        return best_thresholds

    if optimize_threshold:
        thresholds = find_best_thresholds(y_test_d, y_pred_d)
    else:
        thresholds = [0.3, 0.35, 0.5]  

    print(f"Optimized thresholds: {thresholds}")

    y_pred_adjusted = np.zeros_like(y_pred_d)
    for i, threshold in enumerate(thresholds):
        y_pred_adjusted[:, i] = (y_pred_d[:, i] >= threshold).astype(int)
    y_pred = np.argmax(y_pred_adjusted, axis=1)

    y_test = np.argmax(y_test_d, axis=1)  
    print('Accuracy: %.6f' % metrics.accuracy_score(y_test, y_pred))
    print('Precision: %.6f' % metrics.precision_score(y_test, y_pred, average='micro'))
    print('Recall: %.6f' % metrics.recall_score(y_test, y_pred, average='micro'))
    print('F1 Score: %.6f' % metrics.f1_score(y_test, y_pred, average='micro'))
    print('Hamming Loss: %.6f' % metrics.hamming_loss(y_test, y_pred))
    print('Jaccard Score: %.6f' % metrics.jaccard_score(y_test, y_pred, average='micro'))

    unique_classes = np.unique(y_test)
    y_test_bin = y_test_d  
    y_pred_bin = y_pred_d  
    n_classes = len(unique_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    print('Micro-average AUC Score: %.6f' % roc_auc["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC curve (AUC = {0:0.6f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = plt.cm.get_cmap('nipy_spectral', n_classes)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=lw,
                 label='ROC curve of class {0} (AUC = {1:0.6f})'.format(unique_classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multiclass')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_multiclass.png')  
    plt.close()
    print("ROC Curve saved as 'roc_curve_multiclass.png'")

    titles_options = [
        ("Confusion matrix, without normalization", False, 'd'),
        ("Normalized confusion matrix", True, '.2f'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18.5, 8))
    classes = unique_classes

    for ax, (title, normalize, fmt) in zip(axes.flat, titles_options):
        cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, cmap='Blues', fmt=fmt,
                    xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title(title)

    plt.savefig('confusion_matrix_multiclass.png')  
    plt.close()
    print("Confusion Matrix saved as 'confusion_matrix_multiclass.png'")


    print(f"Actual classes in y_test: {unique_classes}")
    classification_rep = metrics.classification_report(
        y_test, y_pred, labels=unique_classes,
        target_names=[str(c) for c in unique_classes],
        zero_division=1, digits=5
    )
    print(classification_rep)





def aggregator_process():
    test_data_path = '~/Documents/Thesis/FL-DT-Classification/Agg/TEST_SAMPLE_ORIGINAL4.csv'
    num_clients = 3  

    client_weights = receive_client_weights(num_clients=num_clients)

    global_weights = federated_averaging(client_weights)

    test_data = pd.read_csv(test_data_path)
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)
    
    X_test = test_data.drop(columns='Attack')
    y_test = test_data['Attack']
    y_test_new =to_categorical(y_test,num_classes=3)

    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(5,)),
        tf.keras.layers.Dense(16, activation=tf.nn.relu),
        #tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(8, activation=tf.nn.relu),
        #tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dense(3, activation=tf.nn.softmax),
    ])

    model.set_weights(global_weights)

    y_pred = model.predict(X_test)
    Performance(X_test, y_test_new, y_pred)


if __name__ == "__main__":
    monitoring_thread = threading.Thread(target=monitor_resources, args=("aggregator_resource_usage.csv",), daemon=True)
    monitoring_thread.start()

    aggregator_process()

    plot_resource_usage("aggregator_resource_usage.csv")
