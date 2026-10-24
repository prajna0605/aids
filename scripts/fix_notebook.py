"""
Complete fix and analyzer for ecg-heartbeat-categorization-using-cnn.ipynb
Addresses all 10 identified issues:
1. Windows Unicode escape error on file paths (use raw strings or relative path with fallback)
2. Kaggle-specific hardcoded paths (fallback to data/mitbih_train.csv and data/mitbih_test.csv)
3. Seaborn 0.13 TypeError: 'int' object is not iterable on x=187
4. Off-by-one slice error :186 (only 186 points) -> changed to :187 (all 187 samples)
5. Missing Python 3.14 support for TensorFlow/Keras -> Native PyTorch 1D-CNN implementation matching the exact architecture
6. Inconsistent reshape for 1D-CNN -> standardized to (N, 1, 187) for Conv1D
7. Accuracy curve plotting error: Y-axis labelled 'Loss' -> corrected to 'Accuracy', plus Loss plot
8. Heatmap numbers displayed in scientific notation -> fixed with fmt='d'
9. Missing classification report and clinical class names
10. Dynamic sampling bound on class 0 to prevent IndexError/ValueError
"""

import json
import shutil
from pathlib import Path

NOTEBOOK_PATH = Path("ecg-heartbeat-categorization-using-cnn.ipynb")
BACKUP_PATH = Path("ecg-heartbeat-categorization-using-cnn.ipynb.bak")

def update_notebook():
    if not NOTEBOOK_PATH.exists():
        print(f"Error: {NOTEBOOK_PATH} not found!")
        return False

    # Create backup
    shutil.copy(NOTEBOOK_PATH, BACKUP_PATH)
    print(f"Created backup at {BACKUP_PATH}")

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Cell 1: Libraries
    cells[1]["source"] = [
        "# importing necessary libraries\n",
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score\n",
        "\n",
        "# Plotting configuration\n",
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
        "class_names = ['0: Normal', '1: PAC', '2: PVC', '3: Fusion', '4: Paced/Unknown']\n"
    ]

    # Cell 2: Import dataset with robust path resolution
    cells[2]["source"] = [
        "# importing dataset with fallback for local workspace vs Kaggle\n",
        "train_path = 'data/mitbih_train.csv' if os.path.exists('data/mitbih_train.csv') else '../input/heartbeat/mitbih_train.csv'\n",
        "data = pd.read_csv(train_path, header=None)\n",
        "df = pd.DataFrame(data)\n",
        "print(f'Successfully loaded dataset from: {train_path}')\n",
        "print(f'Dataset Shape: {df.shape} (Heartbeats: {df.shape[0]}, Features: {df.shape[1]-1} + 1 Class Label)')\n"
    ]

    # Cell 4: Check missing values
    cells[4]["source"] = [
        "# showing column wise percentage of NaN values they contain\n",
        "total_nulls = df.isna().sum().sum()\n",
        "print(f'Total missing values in dataset: {total_nulls}')\n",
        "if total_nulls == 0:\n",
        "    print('Data is completely clean with 0 null values!')\n"
    ]

    # Cell 6: Imbalanced distribution plot (Fix Seaborn 0.13 TypeError on int)
    cells[6]["source"] = [
        "# Imbalanced class distribution visualization (Fixed for Seaborn 0.13+)\n",
        "plt.figure(figsize=(9, 4))\n",
        "sns.countplot(x=df[187], hue=df[187], palette='viridis', legend=False)\n",
        "plt.title('Imbalanced Heartbeat Class Distribution (Original Dataset)', fontsize=13, fontweight='bold')\n",
        "plt.xlabel('Heartbeat Class Label', fontsize=11)\n",
        "plt.ylabel('Number of Samples', fontsize=11)\n",
        "plt.xticks(ticks=range(5), labels=class_names)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()\n"
    ]

    # Cell 8: Balanced undersampling with dynamic bounds
    cells[8]["source"] = [
        "# Class-wise separation and undersampling\n",
        "class_1 = df[df[187] == 1.0]\n",
        "class_2 = df[df[187] == 2.0]\n",
        "class_3 = df[df[187] == 3.0]\n",
        "class_4 = df[df[187] == 4.0]\n",
        "\n",
        "# Sample class 0 safely without exceeding population\n",
        "sample_n = min(len(df[df[187] == 0.0]), 8000)\n",
        "class_0 = df[df[187] == 0.0].sample(n=sample_n, random_state=42)\n",
        "print(f'Sampled {sample_n} heartbeats from Class 0.')\n"
    ]

    # Cell 11: Balanced countplot (Fix Seaborn 0.13 TypeError on int)
    cells[11]["source"] = [
        "# Visualizing balanced class distribution\n",
        "plt.figure(figsize=(9, 4))\n",
        "sns.countplot(x=new_df[187], hue=new_df[187], palette='Set2', legend=False)\n",
        "plt.title('Balanced Heartbeat Class Distribution (After Undersampling)', fontsize=13, fontweight='bold')\n",
        "plt.xlabel('Heartbeat Class Label', fontsize=11)\n",
        "plt.ylabel('Number of Samples', fontsize=11)\n",
        "plt.xticks(ticks=range(5), labels=class_names)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()\n"
    ]

    # Cell 13: Visualizing waveforms (Fix off-by-one slice :186 -> :187 and add labels)
    cells[13]["source"] = [
        "# Plot random sample waveform for each heartbeat class (Full 187 sample points)\n",
        "fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 3))\n",
        "colors = ['#10b981', '#3b82f6', '#ef4444', '#f59e0b', '#8b5cf6']\n",
        "\n",
        "for i in range(5):\n",
        "    sample_df = new_df[new_df[187] == float(i)]\n",
        "    if len(sample_df) > 0:\n",
        "        # Columns 0 to 186 contain the 187 waveform samples\n",
        "        waveform = sample_df.sample(1, random_state=42).iloc[0, :187].values\n",
        "        ax[i].plot(waveform, color=colors[i], linewidth=2)\n",
        "        ax[i].set_title(class_names[i], fontsize=12, fontweight='bold')\n",
        "        ax[i].set_xlabel('Sample Index (0-186)')\n",
        "        ax[i].set_ylabel('Normalized Voltage')\n",
        "        ax[i].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]

    # Cell 14: Train test split with stratify and random_state
    cells[14]["source"] = [
        "# Split data into train and test sets with stratification\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X = new_df.drop([187], axis=1).values\n",
        "y = new_df[187].values.astype(int)\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.1, random_state=42, stratify=y\n",
        ")\n",
        "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')\n"
    ]

    # Cell 15: Reshape for 1D CNN: (batch_size, 1 channel, 187 length)
    cells[15]["source"] = [
        "# Reshape for 1D Convolutional Neural Network: (samples, channels, length)\n",
        "X_train = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])\n",
        "X_test = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])\n",
        "print(f'X_train reshaped for 1D-CNN: {X_train.shape}')\n",
        "print(f'X_test reshaped for 1D-CNN:  {X_test.shape}')\n"
    ]

    # Cell 16: PyTorch 1D-CNN Setup
    cells[16]["source"] = [
        "# 1D-CNN Engine: PyTorch with native Python 3.14 Windows support\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import TensorDataset, DataLoader\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Using compute device: {device}')\n"
    ]

    # Cell 17: CNN Model Definition matching exact layers
    cells[17]["source"] = [
        "# Define 1D-CNN Architecture\n",
        "# Conv1D(32, k=3) -> Conv1D(64, k=3) -> Conv1D(128, k=5) -> MaxPool(k=3, s=2) -> Dropout(0.5) -> Dense(512) -> Dense(1024) -> Dense(5)\n",
        "class ECG_CNN1D(nn.Module):\n",
        "    def __init__(self, num_classes=5):\n",
        "        super(ECG_CNN1D, self).__init__()\n",
        "        # Conv1: 32 filters, kernel 3, padding 'same' (p=1)\n",
        "        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)\n",
        "        self.relu1 = nn.ReLU()\n",
        "        # Conv2: 64 filters, kernel 3, padding 'same' (p=1)\n",
        "        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)\n",
        "        self.relu2 = nn.ReLU()\n",
        "        # Conv3: 128 filters, kernel 5, padding 'same' (p=2)\n",
        "        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)\n",
        "        self.relu3 = nn.ReLU()\n",
        "        # MaxPool with stride 2 and padding 1\n",
        "        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)\n",
        "        self.dropout = nn.Dropout(0.5)\n",
        "        # Linear layers (187 pooled with stride 2 -> 94 length -> 128*94 = 12032)\n",
        "        self.fc1 = nn.Linear(128 * 94, 512)\n",
        "        self.relu_fc1 = nn.ReLU()\n",
        "        self.fc2 = nn.Linear(512, 1024)\n",
        "        self.relu_fc2 = nn.ReLU()\n",
        "        self.out = nn.Linear(1024, num_classes)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.relu1(self.conv1(x))\n",
        "        x = self.relu2(self.conv2(x))\n",
        "        x = self.relu3(self.conv3(x))\n",
        "        x = self.pool(x)\n",
        "        x = self.dropout(x)\n",
        "        x = x.view(x.size(0), -1)\n",
        "        x = self.relu_fc1(self.fc1(x))\n",
        "        x = self.relu_fc2(self.fc2(x))\n",
        "        x = self.out(x)\n",
        "        return x\n",
        "\n",
        "clf = ECG_CNN1D(num_classes=5).to(device)\n",
        "print(clf)\n"
    ]

    # Cell 18: Criterion & Optimizer
    cells[18]["source"] = [
        "# Loss function and Adam Optimizer\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(clf.parameters(), lr=0.001)\n"
    ]

    # Cell 19: Training Loop (10 Epochs)
    cells[19]["source"] = [
        "# Prepare PyTorch DataLoaders & Train for 10 Epochs\n",
        "X_train_t = torch.tensor(X_train, dtype=torch.float32)\n",
        "y_train_t = torch.tensor(y_train, dtype=torch.long)\n",
        "train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)\n",
        "\n",
        "X_test_t = torch.tensor(X_test, dtype=torch.float32)\n",
        "y_test_t = torch.tensor(y_test, dtype=torch.long)\n",
        "test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)\n",
        "\n",
        "epochs = 10\n",
        "history_acc = []\n",
        "history_loss = []\n",
        "\n",
        "print('Starting 1D-CNN Training...')\n",
        "for epoch in range(1, epochs + 1):\n",
        "    clf.train()\n",
        "    total_loss, correct, total = 0.0, 0, 0\n",
        "    for batch_x, batch_y in train_loader:\n",
        "        batch_x, batch_y = batch_x.to(device), batch_y.to(device)\n",
        "        optimizer.zero_grad()\n",
        "        outputs = clf(batch_x)\n",
        "        loss = criterion(outputs, batch_y)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        \n",
        "        total_loss += loss.item() * batch_x.size(0)\n",
        "        _, preds = torch.max(outputs, 1)\n",
        "        correct += (preds == batch_y).sum().item()\n",
        "        total += batch_y.size(0)\n",
        "        \n",
        "    epoch_acc = correct / total\n",
        "    epoch_loss = total_loss / total\n",
        "    history_acc.append(epoch_acc)\n",
        "    history_loss.append(epoch_loss)\n",
        "    print(f'Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%')\n",
        "\n",
        "print('Training complete!')\n"
    ]

    # Cell 20: Validation Inference
    cells[20]["source"] = [
        "# Validation Inference\n",
        "clf.eval()\n",
        "val_preds = []\n",
        "with torch.no_grad():\n",
        "    for batch_x, _ in test_loader:\n",
        "        batch_x = batch_x.to(device)\n",
        "        outputs = clf(batch_x)\n",
        "        _, preds = torch.max(outputs, 1)\n",
        "        val_preds.extend(preds.cpu().numpy())\n",
        "\n",
        "y_lbl = np.array(val_preds)\n"
    ]

    # Cell 21: Training curves (Fix Y-axis label: 'Loss' -> 'Accuracy')
    cells[21]["source"] = [
        "# Plot Training Accuracy & Loss History (Fixed Y-axis labels)\n",
        "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n",
        "epochs_range = range(1, len(history_acc) + 1)\n",
        "\n",
        "# Accuracy subplot\n",
        "ax[0].plot(epochs_range, history_acc, color='#3b82f6', marker='o', linewidth=2, label='Training Accuracy')\n",
        "ax[0].set_title('Training Accuracy Curve', fontsize=12, fontweight='bold')\n",
        "ax[0].set_xlabel('Epochs')\n",
        "ax[0].set_ylabel('Accuracy')\n",
        "ax[0].grid(True, alpha=0.3)\n",
        "ax[0].legend()\n",
        "\n",
        "# Loss subplot\n",
        "ax[1].plot(epochs_range, history_loss, color='#ef4444', marker='s', linewidth=2, label='Training Loss')\n",
        "ax[1].set_title('Training Loss Curve', fontsize=12, fontweight='bold')\n",
        "ax[1].set_xlabel('Epochs')\n",
        "ax[1].set_ylabel('Cross-Entropy Loss')\n",
        "ax[1].grid(True, alpha=0.3)\n",
        "ax[1].legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]

    # Cell 22: Confusion Matrix Heatmap
    cells[22]["source"] = [
        "# Validation Confusion Matrix Heatmap\n",
        "mat = confusion_matrix(y_test, y_lbl)\n",
        "plt.figure(figsize=(7, 6))\n",
        "sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)\n",
        "plt.title('Validation Confusion Matrix', fontsize=12, fontweight='bold')\n",
        "plt.xlabel('Predicted Heartbeat Class')\n",
        "plt.ylabel('Actual Heartbeat Class')\n",
        "plt.show()\n"
    ]

    # Cell 23: Quantitative Validation Metrics
    cells[23]["source"] = [
        "# Measure the Accuracy Score and Classification Report\n",
        "print('=== Validation Performance ===')\n",
        "print(f'Accuracy score of the predictions: {accuracy_score(y_test, y_lbl) * 100:.2f}%')\n",
        "print('\\nDetailed Classification Report:')\n",
        "print(classification_report(y_test, y_lbl, target_names=class_names))\n"
    ]

    # Cell 26: Load test dataset with fallback
    cells[26]["source"] = [
        "# Load test dataset with fallback for local workspace vs Kaggle\n",
        "test_path = 'data/mitbih_test.csv' if os.path.exists('data/mitbih_test.csv') else '../input/heartbeat/mitbih_test.csv'\n",
        "test_data = pd.read_csv(test_path, header=None)\n",
        "test_df = pd.DataFrame(test_data)\n",
        "print(f'Successfully loaded test dataset from: {test_path}')\n",
        "print(f'Test Dataset Shape: {test_df.shape}')\n"
    ]

    # Cell 28: Test countplot (Fix Seaborn 0.13 TypeError on int)
    cells[28]["source"] = [
        "# Test dataset class distribution (Fixed for Seaborn 0.13+)\n",
        "plt.figure(figsize=(9, 4))\n",
        "sns.countplot(x=test_df[187], hue=test_df[187], palette='magma', legend=False)\n",
        "plt.title('Test Dataset Class Distribution', fontsize=13, fontweight='bold')\n",
        "plt.xlabel('Heartbeat Class Label')\n",
        "plt.ylabel('Sample Count')\n",
        "plt.xticks(ticks=range(5), labels=class_names)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.show()\n"
    ]

    # Cell 29: Test waveform plot (Fix slice :186 -> :187)
    cells[29]["source"] = [
        "# Plot random test sample waveform for each heartbeat class (Full 187 sample points)\n",
        "fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25, 3))\n",
        "for i in range(5):\n",
        "    sample_df = test_df[test_df[187] == float(i)]\n",
        "    if len(sample_df) > 0:\n",
        "        waveform = sample_df.sample(1, random_state=42).iloc[0, :187].values\n",
        "        ax[i].plot(waveform, color=colors[i], linewidth=2)\n",
        "        ax[i].set_title(class_names[i], fontsize=12, fontweight='bold')\n",
        "        ax[i].set_xlabel('Sample Index (0-186)')\n",
        "        ax[i].set_ylabel('Normalized Voltage')\n",
        "        ax[i].grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]

    # Cell 30: Reshape test dataset for 1D CNN
    cells[30]["source"] = [
        "# Prepare test dataset tensors: (samples, channels, length)\n",
        "test_X_raw = test_df.drop([187], axis=1).values\n",
        "test_y_true = test_df[187].values.astype(int)\n",
        "\n",
        "test_X = np.array(test_X_raw).reshape(test_X_raw.shape[0], 1, test_X_raw.shape[1])\n",
        "test_X_t = torch.tensor(test_X, dtype=torch.float32)\n",
        "eval_test_loader = DataLoader(TensorDataset(test_X_t), batch_size=128, shuffle=False)\n",
        "print(f'Test feature shape: {test_X.shape}')\n"
    ]

    # Cell 31: Test predictions
    cells[31]["source"] = [
        "# Run test inference\n",
        "clf.eval()\n",
        "test_preds = []\n",
        "with torch.no_grad():\n",
        "    for (batch_x,) in eval_test_loader:\n",
        "        batch_x = batch_x.to(device)\n",
        "        outputs = clf(batch_x)\n",
        "        _, preds = torch.max(outputs, 1)\n",
        "        test_preds.extend(preds.cpu().numpy())\n",
        "\n",
        "test_lbl_y = np.array(test_preds)\n"
    ]

    # Cell 32: Test Confusion Matrix Heatmap
    cells[32]["source"] = [
        "# Test Set Confusion Matrix Heatmap\n",
        "mat_test = confusion_matrix(test_y_true, test_lbl_y)\n",
        "plt.figure(figsize=(7, 6))\n",
        "sns.heatmap(mat_test, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)\n",
        "plt.title('Test Dataset Confusion Matrix (1D-CNN Performance)', fontsize=12, fontweight='bold')\n",
        "plt.xlabel('Predicted Heartbeat Class')\n",
        "plt.ylabel('True Heartbeat Class')\n",
        "plt.show()\n"
    ]

    # Cell 33: Test Accuracy & Report
    cells[33]["source"] = [
        "# Measure Test Accuracy Score and Comprehensive Report\n",
        "print('=== Final Held-Out Test Evaluation ===')\n",
        "print(f'Accuracy score of the predictions: {accuracy_score(test_y_true, test_lbl_y) * 100:.2f}%')\n",
        "print('\\nTest Classification Report:')\n",
        "print(classification_report(test_y_true, test_lbl_y, target_names=class_names))\n"
    ]

    # Clear stale execution counts and outputs so the notebook opens clean and ready to run
    for c in cells:
        if c.get("cell_type") == "code":
            c["execution_count"] = None
            c["outputs"] = []

    # Write back notebook
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print("Successfully updated ecg-heartbeat-categorization-using-cnn.ipynb with all fixes!")
    return True

if __name__ == "__main__":
    update_notebook()
