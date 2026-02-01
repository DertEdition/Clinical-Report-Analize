import json
import os
import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./medikal_db"
DATASET_FILE = "buyuk_medikal_dataset.json" 

client = chromadb.PersistentClient(path=DB_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

try:
    client.delete_collection("tahlil_bilgileri")
    print("Eski veritabanı temizlendi.")
except:
    pass 

collection = client.create_collection(
    name="tahlil_bilgileri",
    embedding_function=embedding_func
)

if not os.path.exists(DATASET_FILE):
    print(f" HATA: '{DATASET_FILE}' bulunamadı! Önce dataset oluşturmalısın.")
    exit()

with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"'{DATASET_FILE}' yüklendi. {len(dataset)} adet tıbbi kayıt işleniyor...")

ids = []
documents = []
metadatas = []

for veri in dataset:
    ids.append(veri["id"])
    
    metadatas.append({
        "test_adi": veri["test_adi"],
        "kategori": veri.get("kategori", "Genel")
    })
    
    yukseklik_veri = veri.get('yukseklik_anlami', 'Belirtilmemiş')
    if isinstance(yukseklik_veri, list):
        yukseklik_text = "\n- " + "\n- ".join(yukseklik_veri)
    else:
        yukseklik_text = str(yukseklik_veri)

    dusukluk_veri = veri.get('dusukluk_anlami', 'Belirtilmemiş')
    if isinstance(dusukluk_veri, list):
        dusukluk_text = "\n- " + "\n- ".join(dusukluk_veri)
    else:
        dusukluk_text = str(dusukluk_veri)
        
    zengin_metin = f"""
    TEST ADI: {veri['test_adi']}
    KATEGORİ: {veri.get('kategori', 'Belirtilmemiş')}
    
    TANIM:
    {veri['genel_aciklama']}
    
    YÜKSEKLİK ANLAMI (NEDEN YÜKSELİR?):
    {yukseklik_text}
    
    DÜŞÜKLÜK ANLAMI (NEDEN DÜŞER?):
    {dusukluk_text}
    
    NORMAL DEĞER BİLGİSİ:
    {veri['normal_deger_notu']}
    """
    
    documents.append(zengin_metin)

print("Veriler vektörleştiriliyor ve ChromaDB'ye yazılıyor...")

BATCH_SIZE = 10
total_batches = len(ids) // BATCH_SIZE + 1

for i in range(0, len(ids), BATCH_SIZE):
    end = i + BATCH_SIZE
    collection.add(
        ids=ids[i:end],
        documents=documents[i:end],
        metadatas=metadatas[i:end]
    )
    print(f"Paket {i//BATCH_SIZE + 1}/{total_batches} yüklendi.")

print("\n" + "="*50)
print(" IMPORT TAMAMLANDI!")
print("="*50)