import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
import sqlite3
import sqlite_vec
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
import time

print("Fine-tuning CLIP text encoder with frozen image embeddings...")

# if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

class DFDataset(Dataset):    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['text'], row['embedding'], row['product_id']

def load_data_into_dataframe():
    products_db_path = 'data/products.db'
    embeddings_db_path = 'data/embeddings.db'
    
    conn = sqlite3.connect(products_db_path)
    products_df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    
    emb_conn = sqlite3.connect(embeddings_db_path)
    emb_conn.enable_load_extension(True)
    sqlite_vec.load(emb_conn)
    emb_conn.enable_load_extension(False)
    
    emb_cursor = emb_conn.cursor()
    emb_cursor.execute("SELECT rowid, embedding FROM embeddings")
    emb_rows = emb_cursor.fetchall()
    emb_conn.close()
    
    embeddings_data = []
    for rowid, embedding_bytes in emb_rows:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32).copy()
        embeddings_data.append({
            'product_id': rowid,
            'embedding': torch.from_numpy(embedding).float()
        })
    
    embeddings_df = pd.DataFrame(embeddings_data)
    
    merged_df = products_df.merge(embeddings_df, left_on='id', right_on='product_id', how='inner')
    
    def create_text(row):
        text_parts = []
        if pd.notna(row['productDisplayName']):
            text_parts.append(row['productDisplayName'])
        if pd.notna(row['articleType']):
            text_parts.append(row['articleType'])
        if pd.notna(row['baseColor']):
            text_parts.append(row['baseColor'])
        if pd.notna(row['gender']):
            text_parts.append(f"for {row['gender']}")
        return " ".join(text_parts)
    
    merged_df['text'] = merged_df.apply(create_text, axis=1)
    
    final_df = merged_df[['product_id', 'text', 'embedding']].copy()
    return final_df

def freeze_image_encoder(model):
    # Freeze image encoder
    for param in model.visual.parameters():
        param.requires_grad = False
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze transformer layers and text projection
    for n, p in model.named_parameters():
        if (n.startswith("transformer.") or n.startswith("ln_final.") or n.startswith("text_projection")):
            p.requires_grad = True

def contrastive_loss(text_embeddings, image_embeddings, temperature=0.07):
    text_embeddings = text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
    image_embeddings = image_embeddings / (image_embeddings.norm(dim=-1, keepdim=True) + 1e-8)
    
    logits = torch.matmul(text_embeddings, image_embeddings.t()) / temperature
    batch_size = text_embeddings.shape[0]
    labels = torch.arange(batch_size, device=text_embeddings.device)
    
    loss_text_to_image = nn.CrossEntropyLoss()(logits, labels)
    loss_image_to_text = nn.CrossEntropyLoss()(logits.t(), labels)
    
    return (loss_text_to_image + loss_image_to_text) / 2

df = load_data_into_dataframe()
print(f"Loaded {len(df)} products")


# Test sample data
# df = df.sample(n=5000, random_state=1016)

train_val_df, test_df = train_test_split(
    df, 
    test_size=0.2, # 20% for testing
    random_state=1016
)

num_folds = 5
num_epochs = 12
batch_size = 32
learning_rate = 1e-6

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

best_val_loss = float('inf')
best_fold = -1

fold_results = []

for fold, (fold_train_idx, fold_val_idx) in enumerate(kfold.split(train_val_df)):
    print(f"\nFold {fold+1}/{num_folds}")
    
    fold_train_df = train_val_df.iloc[fold_train_idx]
    fold_val_df = train_val_df.iloc[fold_val_idx]
    
    model, _ = clip.load("ViT-B/32", device=DEVICE)
    freeze_image_encoder(model)
    
    train_dataset = DFDataset(fold_train_df)
    val_dataset = DFDataset(fold_val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    best_fold_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        total_train_loss = 0
        
        for batch_idx, (texts, image_embeddings, _) in enumerate(train_loader):
            image_embeddings = image_embeddings.to(DEVICE)
            text_tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
            
            text_embeddings = model.encode_text(text_tokens).float()
            loss = contrastive_loss(text_embeddings, image_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f"    {batch_idx+1}/{len(train_loader)}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for texts, image_embeddings, _ in val_loader:
                image_embeddings = image_embeddings.to(DEVICE)
                text_tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
                text_embeddings = model.encode_text(text_tokens).float()
                loss = contrastive_loss(text_embeddings, image_embeddings)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        epoch_time = time.time() - epoch_start_time
        print(f"  Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f} ({epoch_time:.1f}s)")
        
        if avg_val_loss < best_fold_val_loss:
            best_fold_val_loss = avg_val_loss
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_fold = fold + 1
                output_path = "models/finetuned_clip_text_best.pt"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save(model.state_dict(), output_path)
                print(f"Updated best model")
    
    fold_results.append({
        'fold': fold + 1,
        'best_val_loss': best_fold_val_loss
    })

print("Finishing training\n\n")


avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
print(f"\nAverage val loss: {avg_val_loss:.4f}")

for result in fold_results:
    print(f"Fold {result['fold']} val loss: {result['best_val_loss']:.4f}")
