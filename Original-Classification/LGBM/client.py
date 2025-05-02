import pandas as pd
import pickle
import socket
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import label_binarize
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



def train_lgbm(train_data_path, num_leaves=31, n_estimators=100, learning_rate=0.1):

    train_data = pd.read_csv(train_data_path)

    X_train = train_data.drop(columns='Attack')
    y_train = train_data['Attack']

    lgbm_model = LGBMClassifier(
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    lgbm_model.fit(X_train, y_train)

    return lgbm_model



def train_validation(train_data_path, n_splits=3,
                     num_leaves=31, n_estimators=100, learning_rate=0.1):
    

    train_data = pd.read_csv(train_data_path)



    X = train_data.drop(columns='Attack')
    y = train_data['Attack']



    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )



        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
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




def test_lgbm(model, test_data_path):

    test_data = pd.read_csv(test_data_path)
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)

    X_test = test_data.drop(columns='Attack')
    y_test = test_data['Attack']



    y_pred_prob = model.predict_proba(X_test)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, cm, y_test, y_pred, y_pred_prob



def Performance(X_test, y_test_d, y_pred_d):
    y_test = y_test_d.ravel()  
    y_pred = np.argmax(y_pred_d, axis=1)  

    print('Accuracy: %.6f' % metrics.accuracy_score(y_test, y_pred))
    print('Precision: %.6f' % metrics.precision_score(y_test, y_pred, average='micro'))
    print('Recall: %.6f' % metrics.recall_score(y_test, y_pred, average='micro'))
    print('F1 Score: %.6f' % metrics.f1_score(y_test, y_pred, average='micro'))
    print('Hamming Loss: %.6f' % metrics.hamming_loss(y_test, y_pred))
    print('Jaccard Score: %.6f' % metrics.jaccard_score(y_test, y_pred, average='micro'))



    unique_classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=unique_classes)
    y_pred_d_bin = y_pred_d
    n_classes = len(unique_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred_d_bin[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_bin.ravel(), y_pred_d_bin.ravel())
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
    plt.title('Receiver Operating Characteristic (Multiclass)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_multiclass.png')
    plt.close()
    print("ROC Curve saved as 'roc_curve_multiclass.png' in the current directory.")


    titles_options = [
        ("Confusion matrix, without normalization", False, 'd'),
        ("Normalized confusion matrix", True, '.2f'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18.5, 8))
    classes = unique_classes

    for ax, (title, normalize, fmt) in zip(axes.flat, titles_options):
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, cmap='Blues', fmt=fmt,
                    xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title(title)

    plt.savefig('confusion_matrix_multiclass.png')
    plt.close()
    print("Confusion Matrix saved as 'confusion_matrix_multiclass.png' in the current directory.")


    print(f"Actual classes in y_test: {unique_classes}")
    classification_rep = metrics.classification_report(
        y_test, y_pred, labels=unique_classes,
        target_names=[str(c) for c in unique_classes],
        zero_division=1, digits=5
    )
    print(classification_rep)



# Paths
data_path = '~/Documents/Thesis/Original-Classification/TRAINING_SAMPLE_ORIGINAL4_MERGED_Class.csv'
test_data_path = '~/Documents/Thesis/Original-Classification/TEST_SAMPLE_ORIGINAL4.csv'

# Train the model
if __name__ == "__main__":

    #mean_acc, std_dev, std_error = train_validation(data_path,num_leaves=31, n_estimators=100, learning_rate=0.1)

    train_monitor = ResourceMonitor("train_resource_usage.csv")
    train_monitor.start()
    
    lgbm_model = train_lgbm(
        train_data_path=data_path,
        num_leaves=31,      
        n_estimators=100,    
        learning_rate=0.1   
    )

    train_monitor.stop()

    
    plot_resource_usage("train_resource_usage.csv", "Training")


    test_monitor = ResourceMonitor("test_resource_usage.csv")
    test_monitor.start()
    
    (
        test_accuracy,
        test_precision,
        test_recall,
        test_cm,
        y_test,
        y_pred,
        y_pred_prob
    ) = test_lgbm(lgbm_model, test_data_path)

    test_monitor.stop()

    print(f"LightGBM Model Accuracy on Test Data: {test_accuracy}")
    print(f"Precision on Test Data: {test_precision}")
    print(f"Recall on Test Data: {test_recall}")
    print("Confusion Matrix on Test Data:")
    print(test_cm)


    plot_resource_usage("test_resource_usage.csv", "Testing")

    test_data = pd.read_csv(test_data_path)
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(inplace=True)
    X_test = test_data.drop(columns='Attack')
    y_test_vals = test_data['Attack'].values

    Performance(X_test, y_test_vals, y_pred_prob)
