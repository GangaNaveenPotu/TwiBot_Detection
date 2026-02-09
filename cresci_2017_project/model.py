import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define the Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (self.alpha[targets] if self.alpha is not None else 1) * (1 - pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Define the dataset class
class BotDataset(Dataset):
    def __init__(self, texts, metadata, labels, tokenizer, max_len):
        self.texts = texts
        self.metadata = metadata
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        metadata = self.metadata[item]
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(metadata, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define the model
class BotClassifier(torch.nn.Module):
    def __init__(self, n_classes, metadata_input_shape=6):
        super(BotClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        self.metadata_mlp = torch.nn.Sequential(
            torch.nn.Linear(metadata_input_shape, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2), # Improvement Plan 3.2: Add Dropout in Metadata MLP
            torch.nn.Linear(64, 32)
        )
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size + 32, n_classes)

    def forward(self, input_ids, attention_mask, metadata):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        metadata_output = self.metadata_mlp(metadata)
        combined_output = torch.cat([pooled_output, metadata_output], dim=1)
        output = self.drop(combined_output)
        return self.out(output)

# Function to create data loaders
def create_data_loader(df, tokenizer, max_len, batch_size, sampler=None):
    ds = BotDataset(
        texts=df.text.to_numpy(),
        metadata=df[['followers', 'following', 'tweets', 'verified', 'followers_following_ratio', 'tweets_per_day']].to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0, sampler=sampler)

# Main script
if __name__ == '__main__':
    # Load and preprocess data
    df = pd.read_csv('D:/Capstone/cresci-2017/processed_data.csv')
    df['label'] = df['label'].apply(lambda x: 1 if x == 'bot' else 0)

    train_df = df[df.split == 'train']
    val_df = df[df.split == 'val']
    test_df = df[df.split == 'test']

    scaler = StandardScaler()
    numerical_cols = ['followers', 'following', 'tweets', 'followers_following_ratio', 'tweets_per_day']
    train_df.loc[:, numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    val_df.loc[:, numerical_cols] = scaler.transform(val_df[numerical_cols])
    test_df.loc[:, numerical_cols] = scaler.transform(test_df[numerical_cols])

    # Initialize tokenizer and create data loaders
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    MAX_LEN = 64
    BATCH_SIZE = 16

    # Create WeightedRandomSampler for handling class imbalance (Improvement Plan 2.1)
    class_counts = train_df['label'].value_counts().sort_index()
    num_samples = sum(class_counts)
    labels = train_df['label'].to_numpy()
    class_weights = 1. / class_counts.to_numpy()
    sample_weights = np.array([class_weights[label] for label in labels])
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )

    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE, sampler=sampler)
    val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BotClassifier(n_classes=2, metadata_input_shape=6).to(device)
    
    # Freeze all BERT layers first
    for param in model.bert.parameters():
        param.requires_grad = False

    # Unfreeze the last transformer layer (layer 11) and the pooler
    for name, param in model.bert.named_parameters():
        if name.startswith('encoder.layer.11') or name.startswith('pooler'):
            param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-6, weight_decay=1e-5)
    class_weights = torch.tensor([0.659445583820343, 2.0679333209991455], dtype=torch.float).to(device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=2).to(device)

    # Training loop
    EPOCHS = 5
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_data_loader) * EPOCHS)
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    output_dir = "D:/Capstone/results"
    os.makedirs(output_dir, exist_ok=True)


    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for d in tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            metadata = d["metadata"].to(device)
            metadata = torch.nan_to_num(metadata, nan=0.0) # Priority 2: Add training safety net
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
            outputs = torch.clamp(outputs, min=-10, max=10) # Improvement Plan 1.3
            loss = loss_fn(outputs, labels)

            # Improvement Plan 5.1: Detect NaN during training
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected in training at epoch {epoch + 1}, batch {d}. Skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        history['train_loss'].append(train_loss / len(train_data_loader))
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.bin'))


        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for d in tqdm(val_data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                metadata = d["metadata"].to(device)
                metadata = torch.nan_to_num(metadata, nan=0.0) # Priority 2: Add training safety net
                labels = d["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
                outputs = torch.clamp(outputs, min=-10, max=10) # Improvement Plan 1.3
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        history['val_loss'].append(val_loss / len(val_data_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {history["train_loss"][-1]:.4f}, Train Acc: {history["train_acc"][-1]:.4f}, Train F1: {history["train_f1"][-1]:.4f}, Val Loss: {history["val_loss"][-1]:.4f}, Val Acc: {history["val_acc"][-1]:.4f}, Val F1: {history["val_f1"][-1]:.4f}')
        
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # Plot and save training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history_df['train_loss'], label='train loss')
    plt.plot(history_df['val_loss'], label='val loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 3, 2)
    plt.plot(history_df['train_acc'], label='train accuracy')
    plt.plot(history_df['val_acc'], label='val accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1, 3, 3)
    plt.plot(history_df['train_f1'], label='train f1')
    plt.plot(history_df['val_f1'], label='val f1')
    plt.legend()
    plt.title('F1 Score')
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.show()

    # Evaluation on test set
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for d in tqdm(test_data_loader, desc="Testing"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            metadata = d["metadata"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, metadata=metadata)
            preds = torch.argmax(outputs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1-score: {test_f1:.4f}")

    # Save test results
    test_results = {'test_accuracy': [test_acc], 'test_f1_score': [test_f1]}
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

    # Generate and save confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    print("Training and evaluation complete. Results saved in D:/Capstone/results")