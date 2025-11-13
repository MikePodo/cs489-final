import sqlite3
import sqlite_vec
import csv
import os
import json

def get_product_styles(product_id):
    with open(f'data/original/fashion-dataset/styles/{product_id}.json', 'r') as f:
        return json.load(f)

def load_image_urls():
    images_csv_path = 'data/original/fashion-dataset/images.csv'

    image_urls = {}
    
    with open(images_csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            product_id = int(row['filename'].replace('.jpg', ''))
            image_urls[product_id] = row['link']

    return image_urls

def sqlite_setup():    
    csv_path = 'data/original/fashion-dataset/styles.csv'
    db_path = 'data/products.db'
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if os.path.exists(db_path):
        # Remove if exists
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")
    
    conn = sqlite3.connect(db_path)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            price INTEGER,
            discountedPrice INTEGER,
            gender TEXT,
            masterCategory TEXT,
            subCategory TEXT,
            articleType TEXT,
            baseColor TEXT,
            season TEXT,
            year INTEGER,
            usage TEXT,
            productDisplayName TEXT,
            image_url TEXT
        )
    ''')
    

    print(f"Loading image urls...")
    image_urls = load_image_urls()
    
    print(f"Building db from csv...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        
        for row in csv_reader:
            try:
                styles = get_product_styles(row['id'])
                price = int(styles['data']['price'])
                discountedPrice = int(styles['data']['discountedPrice'])

                if not price:
                    # Skip products with no price
                    continue

                product_id = int(row['id'])
                image_url = image_urls.get(product_id, None)

                cursor.execute('''
                    INSERT INTO products (id, price, discountedPrice, gender, masterCategory, subCategory, 
                                      articleType, baseColor, season, year, usage, 
                                      productDisplayName, image_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    product_id,
                    price,
                    discountedPrice,
                    row['gender'],
                    row['masterCategory'],
                    row['subCategory'],
                    row['articleType'],
                    row['baseColour'],
                    row['season'],
                    int(row['year']) if row['year'] else None,
                    row['usage'],
                    row['productDisplayName'],
                    image_url
                ))
                    
            except Exception as e:
                print(f"Error inserting row {row.get('id')}: {e}")
                continue
    
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM products")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows: {total_rows}")
    
    conn.close()


sqlite_setup()
