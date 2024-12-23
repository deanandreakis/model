from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

csv_file = 'ads_creative_text_sample.csv'
data = pd.read_csv(csv_file)

features = data['dimensions'].values
labels = data['text'].values

# Encode string labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Example of converting string features to a one-hot encoding
# Here we convert each string feature to a list of integers
# This may vary based on your specific data and use case
feature_set = list(set(features)) # Get unique features
feature_encoder = LabelEncoder()
encoded_features = feature_encoder.fit_transform(features)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert feature to appropriate format, e.g., one-hot vector
        feature = torch.tensor(self.features[idx]).float()  # Adjust dtype if necessary
        label = torch.tensor(self.labels[idx]).long()
        return feature, label

# Create the dataset
dataset = CustomDataset(encoded_features, encoded_labels)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_features, batch_labels in dataloader:
    # Process the batches here
    print(batch_features.shape, batch_labels.shape)
