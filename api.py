from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import re
import pandas as pd
import pdfplumber
import chromadb
import ollama
from ollama import Client
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tempfile

load_dotenv(override=True)

# Ollama client - Docker için OLLAMA_HOST ortam değişkenini kullanır
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = Client(host=OLLAMA_HOST)

app = FastAPI(title="Medikal Analiz API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_MODEL = "llama3.2"

# Response Model
class AIAnalysisResult(BaseModel):
    status: str
    message: str
    anormallikler: List[str]
    rapor: str
    tablo_sayisi: int = Field(alias="tablo_sayisi")

    class Config:
        populate_by_name = True


def veritabani_baglan():
    try:
        client = chromadb.PersistentClient(path="./medikal_db")
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        return client.get_collection("tahlil_bilgileri", embedding_function=embedding_func)
    except:
        return None


def tahlil_analiz_motoru(pdf_path: str):
    """
    PDF dosyasından tahlil sonuçlarını analiz eder.
    """
    anormallikler = []
    tum_veriler = []
    eklenen_testler = set()

    YASAKLI_BIRIMLER = [
        "g/dL", "mg/dL", "ug/dL", "uL", "IU/L", "U/L", "%", "ng/mL", 
        "mm/h", "fL", "pg", "deg", "10^3/uL", "10^6/uL", "mU/L", "None", "nan",
        "Sonuç", "Referans", "Birim", "Değer"
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            for table in tables:
                if table:
                    tum_veriler.append(pd.DataFrame(table))
                
                for row in table:
                    if not row: continue

                    row = [str(hucre).strip() if hucre else "" for hucre in row]  
                    
                    ref_min, ref_max, ref_index = None, None, -1
                    
                    for i, cell in enumerate(row):
                        match = re.search(r'(\d+[.,]?\d*)\s*-\s*(\d+[.,]?\d*)', cell)
                        if match:
                            try:
                                ref_min = float(match.group(1).replace(',', '.'))
                                ref_max = float(match.group(2).replace(',', '.'))
                                ref_index = i
                                break
                            except: continue

                    if ref_min is None: continue
                    
                    sonuc = None
                    sonuc_index = -1
                    
                    for i in range(ref_index - 1, -1, -1):
                        cell = row[i]
                        
                        if re.search(r'[a-zA-Z]', cell):
                            continue
                        
                        temiz_hucre = re.sub(r'[^\d.,]', '', cell)
                        if temiz_hucre:
                            try:
                                val = float(temiz_hucre.replace(',', '.'))
                                if val < ref_max * 50: 
                                    sonuc = val
                                    sonuc_index = i
                                    break 
                            except: pass
                    
                    test_adi = ""
                    for i in range(sonuc_index - 1, -1, -1):
                        cell_text = row[i].strip()
                        
                        if len(cell_text) < 2: continue
                        if re.match(r'^[\d.,]+$', cell_text): continue
                        
                        if cell_text in YASAKLI_BIRIMLER:
                            continue 
                        is_unit = False
                        for unit in YASAKLI_BIRIMLER:
                            if unit in cell_text and len(cell_text) < len(unit) + 3:
                                is_unit = True
                                break
                        if is_unit: continue

                        test_adi = cell_text
                        break

                    if sonuc is not None and test_adi:
                        clean_name = test_adi.lower().strip()
                        if clean_name in eklenen_testler:
                            continue 
                        
                        durum = None
                        if sonuc < ref_min: durum = "Düşük"
                        elif sonuc > ref_max: durum = "Yüksek"
                        
                        if durum:
                            anormallikler.append({
                                "test_adi": test_adi,
                                "sonuc": str(sonuc),
                                "referans": f"{ref_min} - {ref_max}",
                                "durum": durum
                            })
                            eklenen_testler.add(clean_name)
                            
    return tum_veriler, anormallikler


def rapor_yaz(anormallikler, collection):
    if not collection: 
        return "Veritabanı bağlantısı yok."
    
    def metni_akilli_filtrele(ham_metin, hasta_durumu):
        satirlar = ham_metin.split('\n')
        filtrelenmis_metin = ""
        
        if "Yüksek" in hasta_durumu:
            baslangic_kelimesi = "YÜKSEKLİK ANLAMI"
            bitis_kelimesi = "DÜŞÜKLÜK ANLAMI"
        elif "Düşük" in hasta_durumu:
            baslangic_kelimesi = "DÜŞÜKLÜK ANLAMI"
            bitis_kelimesi = "NORMAL DEĞER"
        else:
            return ham_metin 
        
        kayit_basladi = False
        for satir in satirlar:
            if baslangic_kelimesi in satir:
                kayit_basladi = True
                filtrelenmis_metin += "**OLASI SEBEPLER:**\n" 
                continue
            
            if bitis_kelimesi in satir:
                kayit_basladi = False
                break
            
            if kayit_basladi:
                filtrelenmis_metin += satir + "\n"
                
        return filtrelenmis_metin if len(filtrelenmis_metin) > 10 else ham_metin

    context_data = ""
    for bulgu in anormallikler:
        results = collection.query(
            query_texts=[bulgu['test_adi']],
            n_results=1,
            where={"test_adi": bulgu['test_adi']} 
        )
        
        if not results['documents'] or not results['documents'][0]:
            results = collection.query(query_texts=[bulgu['test_adi']], n_results=1)

        if results['documents'] and results['documents'][0]:
            raw_db_info = results['documents'][0][0]
            ozel_bilgi = metni_akilli_filtrele(raw_db_info, bulgu['durum'])
            
            context_data += f"""
            ### TEST: {bulgu['test_adi']}
            DURUM: {bulgu['durum']} (Sonuç: {bulgu['sonuc']} | Referans: {bulgu['referans']})
            
            {ozel_bilgi}
            --------------------------------------------------
            \n
            """
        else:
            context_data += f"Bilgi bulunamadı: {bulgu['test_adi']}\n"

    system_prompt = """
    Sen uzman bir Türk tıbbi asistanısın. Görevin hasta sonuçlarını analiz edip raporlamaktır.
    
    KURALLAR:
    1. Sana verilen metindeki sebepleri madde madde yaz.
    2. Asla İngilizce kelime kullanma.
    3. Çıktı formatın şöyle olsun:
       
       **[Test Adı] (Sonuç: [Değer] | Referans: [Aralık])**
       **Durum:** Değeriniz referans aralığının [üzerindedir/altındadır].
       
       **Olası Sebepler:**
       - [Madde 1]
       - [Madde 2]
       
       **Öneri:** Lütfen doktorunuzla görüşünüz.
    """
    
    user_prompt = f"""
    Aşağıdaki temizlenmiş hasta verilerini kullanarak raporu yaz:
    
    {context_data}
    """
    
    try:
        response = ollama_client.chat(
            model=LOCAL_MODEL, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0.1} 
        )
        return response['message']['content']
    except Exception as e:
        return f"Local Model Hatası: {str(e)}"


@app.post("/analyze", response_model=AIAnalysisResult)
async def analyze_blood_test(file: UploadFile = File(...)):
    """
    PDF kan tahlili dosyasını analiz eder ve JSON formatında sonuç döner.
    """
    print(f"[LOG] İstek alındı: {file.filename}")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları kabul edilir.")
    
    print("[LOG] Veritabanına bağlanılıyor...")
    # Veritabanı bağlantısı
    collection = veritabani_baglan()
    print(f"[LOG] Veritabanı bağlantısı: {'OK' if collection else 'HATA'}")
    if not collection:
        return AIAnalysisResult(
            status="error",
            message="Veritabanı (medikal_db) bulunamadı! 'import_dataset.py' çalıştırın.",
            anormallikler=[],
            rapor="",
            tablo_sayisi=0
        )
    
    # Geçici dosyaya kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Analiz yap
        tablolar, sorunlar = tahlil_analiz_motoru(tmp_path)
        
        if not sorunlar:
            return AIAnalysisResult(
                status="success",
                message="Tüm değerler referans aralığında.",
                anormallikler=[],
                rapor="",
                tablo_sayisi=len(tablolar)
            )
        
        # Anormallikleri string listesine çevir
        anormallik_listesi = [
            f"{s['test_adi']}: {s['sonuc']} ({s['durum']}) - Referans: {s['referans']}"
            for s in sorunlar
        ]
        
        # Yapay zeka raporu oluştur
        rapor = rapor_yaz(sorunlar, collection)
        
        return AIAnalysisResult(
            status="success",
            message=f"{len(sorunlar)} adet referans dışı değer bulundu.",
            anormallikler=anormallik_listesi,
            rapor=rapor,
            tablo_sayisi=len(tablolar)
        )
        
    finally:
        # Geçici dosyayı sil
        os.unlink(tmp_path)


@app.get("/health")
async def health_check():
    """Servis sağlık kontrolü"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
