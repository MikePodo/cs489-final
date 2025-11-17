import torch
import clip
import sqlite3
import sqlite_vec
import numpy as np
import time

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


base_model, _ = clip.load("ViT-B/32", device=DEVICE)
base_model.eval()

finetuned_model, _ = clip.load("ViT-B/32", device=DEVICE)
finetuned_model.load_state_dict(torch.load("models/finetuned_clip_text_best.pt"))
finetuned_model.eval()

products_conn = sqlite3.connect('data/products.db')
products_conn.row_factory = sqlite3.Row

embeddings_conn = sqlite3.connect('data/embeddings.db')
embeddings_conn.enable_load_extension(True)
sqlite_vec.load(embeddings_conn)
embeddings_conn.enable_load_extension(False)

def search_products(query_text, model, limit=5):
    start_time = time.time()

    # Encode query text
    with torch.no_grad():
        text_tokens = clip.tokenize([query_text]).to(DEVICE)
        query_embedding = model.encode_text(text_tokens).float()
        query_embedding = query_embedding.squeeze().cpu().numpy().astype(np.float32)
    
    cursor = embeddings_conn.cursor()
    cursor.execute("""
        SELECT rowid, distance
        FROM embeddings
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (query_embedding.tobytes(), limit))
    
    results = []
    prices = []
    for rowid, distance in cursor.fetchall():
        prod_cursor = products_conn.cursor()
        prod_cursor.execute("SELECT * FROM products WHERE id = ?", (rowid,))
        product = prod_cursor.fetchone()
        if not product:
            continue

        price = product['discountedPrice'] if product['discountedPrice'] else product['price']
        results.append({
            'id': rowid,
            'name': product['productDisplayName'],
            "image_url": product['image_url'],
            'distance': distance,
            'price': price
        })
        if price:
            prices.append(price)
    
    # Sort with price consideration
    if prices:
        max_price = max(prices)
        for result in results:
            if result['price']:
                price_factor = (result['price'] / max_price) * 0.1
                result['score'] = result['distance'] + price_factor
            else:
                result['score'] = result['distance']
        
        results.sort(key=lambda x: x['score'])
    
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    return results, duration_ms

def print_results(results):
    for i, result in enumerate(results, 1):
        print(f"{i}.\n  Name: {result['name']}\n  Price: ${result['price'] / 100:.2f}\n  Image: {result['image_url']}\n  Distance: {result['distance']:.4f}")

query = input("\nEnter query: ").strip()

base_results, base_time = search_products(query, base_model, limit=5)
finetuned_results, finetuned_time = search_products(query, finetuned_model, limit=5)

print(f"\nBASE CLIP ({int(base_time)}ms):")
print_results(base_results)
print(f"\nFINE-TUNED CLIP ({int(finetuned_time)}ms):")
print_results(finetuned_results)
    

products_conn.close()
embeddings_conn.close()
