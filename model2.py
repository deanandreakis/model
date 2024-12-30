import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

data = pd.read_csv('ads_creative_text_sample.csv')
ads = [str(i) for i in data['text'].tolist()]#data['text'].values
sizes = data['dimensions'].values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

tokenized_ads = tokenize_text(ads)

size_to_id = {size: idx for idx, size in enumerate(set(sizes))}
size_ids = [size_to_id[size] for size in sizes]

class AdDataset(Dataset):
       def __init__(self, tokenized_ads, sizes):
           self.tokenized_ads = tokenized_ads
           self.sizes = sizes

       def __len__(self):
           return len(self.sizes)

       def __getitem__(self, idx):
           ad = {key: val[idx] for key, val in self.tokenized_ads.items()}
           size = torch.tensor(self.sizes[idx])
           return ad, size

dataset = AdDataset(tokenized_ads, size_ids)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

num_epochs = 2
device = torch.device("mps")

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

model.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        inputs = batch[0]  # Input text
        targets = inputs['input_ids'].to(device)
        sizes = batch[1].to(device)

        # Forward pass
        outputs = model(**inputs, labels=targets)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

