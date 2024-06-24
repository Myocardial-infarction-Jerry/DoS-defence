import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
from config import MODEL_PARAMS, CHECKPOINT_DIR, get_checkpoint_path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """Load data into a pandas DataFrame"""
    logging.info(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    logging.info(f"Data loaded with shape: {data.shape}")
    return data


def preprocess_labels(labels):
    """Convert string labels to integer labels"""
    label_map = {
        'normal': 0,
        'neptune': 1,
        'portsweep': 1,
        'satan': 1,
        'ipsweep': 1,
        'smurf': 1,
        'back': 1,
        'guess_passwd': 1,
        'teardrop': 1,
        'pod': 1,
        'nmap': 1,
        'warezclient': 1,
        'land': 1,
        'ftp_write': 1,
        'multihop': 1,
        'rootkit': 1,
        'buffer_overflow': 1,
        'imap': 1,
        'warezmaster': 1,
        'phf': 1,
        'saint': 1,
        'mscan': 1,
        'apache2': 1,
        'httptunnel': 1,
        'xterm': 1,
        'ps': 1,
        'sqlattack': 1,
        'snmpgetattack': 1,
        'snmpguess': 1,
        'mailbomb': 1,
        'named': 1,
        'sendmail': 1,
        'xsnoop': 1,
        'worm': 1,
        'spy': 1,
        'loadmodule': 1,
        'perl': 1,
        'xlock': 1,
        'xsan': 1,
        'udplag': 1,
        'processtable': 1,
        'sql': 1
        # Add other attack labels here if necessary
    }
    return labels.map(label_map).fillna(0).astype('int32')


def preprocess_features(train_data, test_data):
    """Convert categorical features to numerical using one-hot encoding and ensure both datasets have the same columns"""
    categorical_columns = ['protocol_type', 'service', 'flag']
    combined_data = pd.concat([train_data, test_data], keys=['train', 'test'])
    combined_data = pd.get_dummies(combined_data, columns=categorical_columns)
    train_data = combined_data.xs('train')
    test_data = combined_data.xs('test')
    return train_data, test_data


def plot_confusion_matrix(cm, classes, filename='confusion_matrix.png', title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot and save confusion matrix to a file"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(filename)
    plt.close()
    logging.info(f"Confusion matrix plot saved as {filename}")


def plot_roc_curve(fpr, tpr, roc_auc, filename='roc_curve.png'):
    """Plot ROC curve and save to a file"""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.savefig(filename)
    plt.close()
    logging.info(f"ROC curve plot saved as {filename}")


# Define file paths for the train and test datasets
train_file = "datasets/kdd_train.csv"
test_file = "datasets/kdd_test.csv"

# Load datasets
train_data = load_data(train_file)
test_data = load_data(test_file)

# Preprocess labels
train_data['labels'] = preprocess_labels(train_data['labels'])
test_data['labels'] = preprocess_labels(test_data['labels'])

# Preprocess features
train_data, test_data = preprocess_features(train_data, test_data)

# Select features and labels
X_train = train_data.drop('labels', axis=1)
y_train = train_data['labels']

X_test = test_data.drop('labels', axis=1)
y_test = test_data['labels']

# Convert all columns to float32 as required by scikit-learn
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Get the checkpoint path based on model parameters
checkpoint_path = get_checkpoint_path(MODEL_PARAMS)

# Initialize the model
model = RandomForestClassifier(**MODEL_PARAMS)

# Check if checkpoint exists
if os.path.exists(checkpoint_path):
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    model = joblib.load(checkpoint_path)
else:
    # Train the model with tqdm progress bar
    logging.info("Initializing and training the model...")
    for i in tqdm(range(model.n_estimators), desc="Training Progress"):
        model.n_estimators = i + 1
        model.fit(X_train, y_train)
    logging.info("Model training completed.")
    # Save checkpoint
    joblib.dump(model, checkpoint_path)

# Predictions
logging.info("Making predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
logging.info(f'Accuracy: {accuracy:.2f}')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plot_roc_curve(fpr, tpr, roc_auc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['Normal', 'Attack'])

logging.info("Process completed.")
