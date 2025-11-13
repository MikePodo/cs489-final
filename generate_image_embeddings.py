import torch
import clip
from PIL import Image
import sqlite3
import sqlite_vec
import os
import numpy as np

print("Generating image embeddings...")

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

model, preprocess = clip.load("ViT-B/32", device=DEVICE)

embedding_db_path = 'data/embeddings.db'
os.makedirs(os.path.dirname(embedding_db_path), exist_ok=True)
if os.path.exists(embedding_db_path):
    # Remove if exists
    os.remove(embedding_db_path)
    print(f"Removed existing database at {embedding_db_path}")
    
embedding_conn = sqlite3.connect(embedding_db_path)

embedding_conn.enable_load_extension(True)
sqlite_vec.load(embedding_conn)
embedding_conn.enable_load_extension(False)

embedding_cursor = embedding_conn.cursor()

embedding_cursor.execute('''
    CREATE VIRTUAL TABLE embeddings USING vec0(
        embedding float[512]
    )
''')

embedding_conn.commit()

db_path = 'data/products.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT * FROM products")
rows = cursor.fetchall()

print(f"\nProcessing {len(rows)} products...")

for idx, row in enumerate(rows):
    try:
        product_id = row['id']
        
        image_path = f'data/original/fashion-dataset/images/{product_id}.jpg'
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            image_embedding = model.encode_image(image)
        
        embedding_array = image_embedding.squeeze().cpu().detach().numpy().astype(np.float32)
        
        embedding_cursor.execute(
            "INSERT INTO embeddings(rowid, embedding) VALUES (?, ?)",
            (product_id, embedding_array.tobytes())
        )
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(rows)} products...")
            embedding_conn.commit()
            
    except Exception as e:
        print(f"Error processing {product_id}: {e}")
        continue

embedding_conn.commit()

conn.close()
embedding_conn.close()
