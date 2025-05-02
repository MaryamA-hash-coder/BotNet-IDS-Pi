# BotNet-IDS-Pi

This project implements a resource-efficient Intrusion Detection System (IDS) for IoT environments. It supports both centralized and federated learning approaches using the BoTNeTIoT-L01 dataset and is tested on 4 Raspberry Pi 4 Model B devices. The objective is to detect and classify malicious behavior (Mirai, Gafgyt) from normal IoT traffic using machine learning models.


| Centralized Models     |
|------------------------|
| `Logistic Regression`  |
| `Decision Tree`        |
| `Random Forest`        |
| `Bagging Classifier`   |
| `AdaBoost`             |
| `Gradient Boosting`    |
| `XGBoost`              |
| `LightGBM`             |
| `Neural Network `      |


| Federated Model   | Aggregation Strategy           |
|-------------------|---------------------------------|
| `Decision Tree`   | **Ensemble Averaging**          |
| `Random Forest`   | **Ensemble Averaging**          |
| `Neural Network`  | **Federated Averaging (FedAvg)**|





---

## Experimental Setup

### Devices
- **4 Raspberry Pi 4 Model B** with:
  - 4GB RAM
  - 64GB microSD card
  - Raspberry Pi OS 64-bit (set up using Raspberry Pi Imager)
  - SSH access enabled for remote control

### Roles
- 1 Raspberry Pi acts as **Aggregator Node**
- 3 Raspberry Pis act as **Federated Clients**

---

## Prerequisites

Ensure the following on **each Raspberry Pi**:

### 1. Create Virtual Environment

```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv myenv
source myenv/bin/activate
```

### 2. Install Python Dependencies
pip install --upgrade pip
pip install tensorflow pandas scikit-learn xgboost lightgbm gputil psutil


---

## Running the System

### 1. Centralized Learning (Run on any one Pi)
This will:
- Preprocess the dataset
- Train and evaluate centralized models (DT, RF, NN, etc.)
- Log performance and resource metrics (CPU, RAM, etc.)

### 2. Federated Learning
a. Aggregator (Run on Aggregator Pi)
The aggregator will:
- Recieve updates from clients via TCP
- Aggregate the model updates
- Test the model (DecisionTree, RF, or NN)
- Log performance and resource metrics (CPU, RAM, etc.)

b. Clients (Run on 3 Client Pis)
The client will:
- Load local data
- Train a model (DecisionTree, RF, or NN)
- Send updates to the aggregator via TCP
- Log performance and resource metrics (CPU, RAM, etc.)

---

##  Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score, AUC
- Hamming Loss, Jaccard Score
- Resource Monitoring:
  - CPU Usage
  - RAM Usage
  - CPU Temperature
  - Disk Read/Write
  - Network Sent/Received

---

## 📊 Dataset

- **BoTNeTIoT-L01**
  - Over 7 million records
  - Preprocessed to select **5 features** using mutual information
  - Highly imbalanced dataset (Mirai, Gafgyt, Normal)









