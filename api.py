from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import re
import pandas as pd
import pdfplumber
import chromadb
from groq import Groq
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tempfile

load_dotenv(override=True)

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Medikal Analiz API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def ingilizce_temizle(metin: str) -> str:
    sozluk = {
        "dehydration": "susuzluk",
        "ruled out": "ekarte edilmesi",
        "rule out": "ekarte etmek",
        "slightly": "hafifçe",
        "possibly": "olasılıkla",
        "possible": "olası",
        "elevated": "yüksek",
        "increased": "artmış",
        "decreased": "azalmış",
        "benign": "zararsız",
        "beberapa": "",
    }
    for ing, tr in sozluk.items():
        metin = re.sub(ing, tr, metin, flags=re.IGNORECASE)
    return metin


def hasta_bilgisi_cek(pdf_path: str) -> dict:
    """
    PDF'in ilk sayfasından hasta bilgilerini (ad, cinsiyet, doğum tarihi, yaş, tarih) çeker.
    """
    bilgi = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""

                # Cinsiyet
                cinsiyet_match = re.search(r'Cinsiyet:\s*(\w+)', text)
                if cinsiyet_match:
                    bilgi['cinsiyet'] = cinsiyet_match.group(1)

                # Doğum tarihi ve yaş hesaplama
                dogum_match = re.search(r'Doğum\s*Tarihi:\s*(\d{2}\.\d{2}\.\d{4})', text)
                tarih_match = re.search(r'Tarih:\s*(\d{2}\.\d{2}\.\d{4})', text)
                if dogum_match:
                    bilgi['dogum_tarihi'] = dogum_match.group(1)
                    if tarih_match:
                        from datetime import datetime
                        dogum = datetime.strptime(dogum_match.group(1), "%d.%m.%Y")
                        rapor_tarih = datetime.strptime(tarih_match.group(1), "%d.%m.%Y")
                        yas = (rapor_tarih - dogum).days // 365
                        bilgi['yas'] = yas
                        bilgi['rapor_tarihi'] = tarih_match.group(1)

                # Ad/Soyad
                ad_match = re.search(r'Adı/Soyadı:\s*(.+?)(?:\s+Cinsiyet|$)', text)
                if ad_match:
                    bilgi['ad_soyad'] = ad_match.group(1).strip()
    except Exception:
        pass
    return bilgi


def tahlil_analiz_motoru(pdf_path: str):
    """
    PDF dosyasından tahlil sonuçlarını analiz eder.
    """
    anormallikler = []
    tum_veriler = []
    eklenen_testler = set()

    YASAKLI_BIRIMLER = [
        "g/dL", "mg/dL", "ug/dL", "uL", "IU/L", "U/L", "%", "ng/mL",
        "mm/h", "fL", "pg", "deg", "10^3/uL", "10^6/uL", "10^9/L", "10^12/L",
        "mU/L", "mL/dk/1.73", "mL/dk", "ng/dL", "ng/L", "ug/L", "mg/L",
        "mmol/L", "None", "nan",
        "Sonuç", "Referans", "Birim", "Değer"
    ]

    # Birim hücrelerini tanımak için regex pattern
    BIRIM_PATTERN = re.compile(
        r'^(?:\d+\^?\d*/[a-zA-Z]|[a-zA-Z]+/[a-zA-Z]+|m[Ll]/dk|[munpf]?[gLl]/[dmu]?[Ll])',
        re.IGNORECASE
    )

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
                        cell = row[i].strip()

                        # Birim hücrelerini atla
                        if cell in YASAKLI_BIRIMLER:
                            continue
                        # 10^3/uL, 10^9/L, 10^12/L, mL/dk/1.73 gibi birimleri atla
                        if re.search(r'10\s*[\^]\s*\d+\s*/\s*[a-zA-Z]', cell):
                            continue
                        if re.search(r'm[Ll]/dk', cell):
                            continue
                        if BIRIM_PATTERN.search(cell):
                            continue
                        # Bilinen birim kalıplarını atla (harf+/+harf içeren kısa hücreler)
                        if re.search(r'[a-zA-Z].*/', cell) and len(cell) < 15:
                            continue

                        # ">" ve "<" karakterlerini temizle
                        temiz_cell = re.sub(r'[<>]', '', cell)

                        # Harf kontrolü: CLASS VI gibi tamamen metin hücreleri atla
                        temiz_hucre = re.sub(r'[^\d.,]', '', temiz_cell)
                        if not temiz_hucre and re.search(r'[a-zA-Z]', cell):
                            continue
                        if temiz_hucre:
                            try:
                                val = float(temiz_hucre.replace(',', '.'))
                                if ref_max > 0 and val < ref_max * 1000:
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
                        # Birim pattern kontrolü
                        if re.search(r'10\s*[\^]\s*\d+\s*/\s*[a-zA-Z]', cell_text):
                            continue
                        if re.search(r'm[Ll]/dk', cell_text):
                            continue
                        if BIRIM_PATTERN.search(cell_text):
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
                        # "Hemogram HGB" ve "HGB" gibi tekrarları önle
                        clean_name = re.sub(r'^hemogram\s+', '', clean_name)
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
    2. Asla İngilizce kelime kullanma. UYARI: 'possibly', 'possible', 'elevated', 'increased' gibi İngilizce kelimeler KESİNLİKLE yasaktır. Tıbbi terimleri de Türkçe karşılıklarıyla yaz (örn: poliglobuli → kırmızı kan hücresi artışı, hiperürisemi → ürik asit yüksekliği).
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
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.1
        )
        return ingilizce_temizle(response.choices[0].message.content)
    except Exception as e:
        return f"Groq API Hatası: {str(e)}"


def sentez_yaz(anormallikler: list, hasta_bilgisi: dict = None) -> str:
    """
    Tüm anormal değerleri birlikte değerlendirerek bütünleşik yorum üretir.
    """
    if not anormallikler:
        return ""

    ozet = "\n".join([
        f"- {b['test_adi']}: {b['sonuc']} ({b['durum']}) | Referans: {b['referans']}"
        for b in anormallikler
    ])

    system_prompt = """
    Sen uzman bir Türk iç hastalıkları doktorusun.
    Sana birden fazla anormal laboratuvar sonucu verilecek.
    Görevin bu sonuçları BERABER değerlendirerek hastaya doğrudan hitap eden bütünleşik bir yorum yazmak.

    KURALLAR:
    1. Hastaya doğrudan "siz/sizin" diye hitap et. "Hasta" kelimesini hiç kullanma.
    2. Değerleri birbirleriyle ilişkilendir. Hangi bulgular birbirini destekliyor veya açıklıyor?
    3. Bu yaş, cinsiyet ve profile göre en sık ve en olası açıklamayı öne çıkar. Ciddi patolojilerden önce zararsız olasılıkları değerlendir.
    4. Asla İngilizce kullanma. Tıbbi terimleri Türkçe karşılıklarıyla yaz.
    5. Spekülatif ol ama ölçülü: "düşündürebilir", "akla getirebilir" gibi ifadeler kullan.
    6. Formatın şöyle olsun:

    ---
    **Bütünleşik Değerlendirme**

    **Öne Çıkan Tablo:**
    [Sizin değerleriniz... diye başla. Bulguların bir arada ne anlama gelebileceğini 2-4 cümleyle açıkla.]

    **Dikkat Edilmesi Gereken İlişkiler:**
    - [Bulgu A] + [Bulgu B] birlikteliği → [Klinik anlam]
    - ...

    **Genel Öneri:**
    [Hastaya doğrudan hitap ederek, alarmist olmadan, bir sonraki adımı belirt.]
    ---
    """

    hasta_ozet = ""
    if hasta_bilgisi:
        parts = []
        if 'yas' in hasta_bilgisi:
            parts.append(f"{hasta_bilgisi['yas']} yaşında")
        if 'cinsiyet' in hasta_bilgisi:
            parts.append(hasta_bilgisi['cinsiyet'].lower())
        if parts:
            hasta_ozet = f"Hasta Bilgisi: {', '.join(parts)} hasta."

    user_prompt = f"""
    {hasta_ozet}

    Anormal laboratuvar sonuçları:
    {ozet}

    NOT: Yukarıda listelenmeyen tüm değerler referans aralığında.
    Bu yaş, cinsiyet ve profilde en olası ve zararsız açıklamayı öne çıkar.
    """

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.2
        )
        return ingilizce_temizle(response.choices[0].message.content)
    except Exception as e:
        return f"Sentez Hatası: {str(e)}"


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
        # Hasta bilgisini çek
        hasta_bilgisi = hasta_bilgisi_cek(tmp_path)
        print(f"[LOG] Hasta bilgisi: {hasta_bilgisi}")

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
        bireysel_rapor = rapor_yaz(sorunlar, collection)
        sentez = sentez_yaz(sorunlar, hasta_bilgisi)
        rapor = bireysel_rapor + "\n\n" + sentez
        
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
