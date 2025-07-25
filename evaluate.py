import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset
from AnomalyTransformer import AnomalyTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_data():
    df = pd.read_csv("datasets/ServerMachineDataset/train/machine-1-1.txt")
    X = df.values
    X = np.pad(X, ((0, 0), (0, 56 - X.shape[1])), mode='constant')
    return X




def calculate_anomaly_score(model, data):
    data_tensor = data.to(next(model.parameters()).device)
    with torch.no_grad():
        anomaly_scores = model.anomaly_score(data_tensor)
    return anomaly_scores

if __name__ == "__main__":
    X= load_data()
    X_train=X
    X_test=pd.read_csv("datasets/ServerMachineDataset/test/machine-1-1.txt")
    X_test=X_test.values
    X_test = np.pad(X_test, ((0, 0), (0, 56 - X_test.shape[1])), mode='constant')
    y_test=pd.read_csv("datasets/ServerMachineDataset/test_label/machine-1-1.txt")
    y_test=y_test.values
    y_test=y_test.reshape(-1)
    N, d_model = X_train.shape[0], X_train.shape[1]
    model = AnomalyTransformer(N, d_model, hidden_dim=64)
    model.load_state_dict(torch.load('anomaly_transformer_weights.pth'))
    model.eval()
    device = next(model.parameters()).device

    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test)), batch_size=1024, shuffle=False)
    anomaly_scores = []
    for data in test_loader:
        data = torch.FloatTensor(data[0]).to(device)
        anomaly_score = calculate_anomaly_score(model, data)
        anomaly_scores.append(anomaly_score)

    anomaly_scores = torch.cat(anomaly_scores, dim=0)



    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(anomaly_scores.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()


    threshold = np.mean(centers)
    predicted_labels = []
    for score in anomaly_scores:
        if score < threshold:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)


    f1 = f1_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    accuracy = accuracy_score(y_test, predicted_labels)
    bce=log_loss(y_test,predicted_labels)
    mse=mean_squared_error(y_test,predicted_labels)

    conf_matrix = confusion_matrix(y_test, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")