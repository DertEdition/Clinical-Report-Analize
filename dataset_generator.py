import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv 

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY bulunamadı")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/models/gemini-2.5-pro")

FILE_NAME = "buyuk_medikal_dataset.json"
MAX_PER_RUN = 20

analysis_list = [
    # --- HEMOGRAM (TAM KAN) ---
    "WBC (Lökosit - Beyaz Kan Hücresi)", 
    "RBC (Eritrosit - Kırmızı Kan Hücresi)", 
    "HGB (Hemoglobin)", 
    "HCT (Hematokrit)", 
    "MCV (Ortalama Eritrosit Hacmi)", 
    "MCH (Ortalama Eritrosit Hemoglobini)", 
    "MCHC (Ortalama Eritrosit Hemoglobin Konsantrasyonu)", 
    "PLT (Trombosit)", 
    "RDW (Eritrosit Dağılım Genişliği)", 
    "MPV (Ortalama Trombosit Hacmi)", 
    "PCT (Trombositokrit)",
    "NEU% (Nötrofil Yüzdesi)", "NEU# (Nötrofil Sayısı)",
    "LYM% (Lenfosit Yüzdesi)", "LYM# (Lenfosit Sayısı)",
    "MONO% (Monosit Yüzdesi)", "MONO# (Monosit Sayısı)",
    "EOS% (Eozinofil Yüzdesi)", "EOS# (Eozinofil Sayısı)",
    "BASO% (Bazofil Yüzdesi)", "BASO# (Bazofil Sayısı)",

    # --- BİYOKİMYA (GENEL) ---
    "Glukoz (Açlık Kan Şekeri)", 
    "Tokluk Kan Şekeri",
    "HbA1c (3 Aylık Şeker)", 
    "İnsülin (Hormon)", 
    "HOMA-IR (İnsülin Direnci)",
    "Üre (BUN)", 
    "Kreatinin", 
    "Ürik Asit", 
    "Total Protein", 
    "Albumin", 
    "Globulin",

    # --- KARACİĞER FONKSİYONLARI ---
    "AST (Aspartat Transaminaz)", 
    "ALT (Alanin Aminotransferaz)", 
    "GGT (Gama Glutamil Transferaz)", 
    "ALP (Alkalen Fosfataz)", 
    "LDH (Laktat Dehidrogenaz)", 
    "Total Bilirubin", 
    "Direkt Bilirubin", 
    "İndirekt Bilirubin",

    # --- ELEKTROLİTLER & MİNERALLER ---
    "Sodyum (Na)", 
    "Potasyum (K)", 
    "Klor (Cl)", 
    "Kalsiyum (Ca)", 
    "Magnezyum (Mg)", 
    "Fosfor (P)", 
    "Çinko (Zn)",

    # --- KALP & LİPİDLER (YAĞLAR) ---
    "Total Kolesterol", 
    "LDL Kolesterol (Kötü)", 
    "HDL Kolesterol (İyi)", 
    "Trigliserit", 
    "CK (Kreatin Kinaz)", 
    "CK-MB", 
    "Troponin",

    # --- TİROİD & HORMONLAR ---
    "TSH (Tiroid Stimülan Hormon)", 
    "Serbest T3", 
    "Serbest T4", 
    "Anti-TPO", 
    "Anti-Tg",
    "FSH (Folikül Uyarıcı Hormon)", 
    "LH (Lüteinleştirici Hormon)", 
    "Prolaktin", 
    "Estradiol (E2)", 
    "Progesteron", 
    "Testosteron (Total)", 
    "Kortisol", 
    "Beta-HCG (Gebelik)",

    # --- VİTAMİNLER & DEMİR ---
    "Demir (Serum)", 
    "Demir Bağlama Kapasitesi (TDBK)", 
    "Ferritin (Demir Deposu)", 
    "Vitamin B12", 
    "Folat (Folik Asit)", 
    "Vitamin D (25-OH)",

    # --- İNFLAMASYON & ROMATİZMA ---
    "CRP (C-Reaktif Protein)", 
    "Sedimantasyon (ESR)", 
    "ASO (Antistreptolizin O)", 
    "RF (Romatoid Faktör)",

    # --- PIHTILAŞMA ---
    "PT (Protrombin Zamanı)", 
    "INR", 
    "aPTT", 
    "Fibrinojen", 
    "D-Dimer",

    # --- TAM İDRAR TAHLİLİ (URINALYSIS) ---
    "İdrar Rengi",
    "İdrar Görünümü (Berraklık)",
    "İdrar pH",
    "İdrar Dansite (Yoğunluk)",
    "İdrarda Protein",
    "İdrarda Glukoz",
    "İdrarda Keton",
    "İdrarda Bilirubin",
    "İdrarda Ürobilinojen",
    "İdrarda Nitrit",
    "İdrarda Lökosit Esteraz",
    "İdrar Mikroskopisi: Lökosit (WBC)",
    "İdrar Mikroskopisi: Eritrosit (RBC)",
    "İdrar Mikroskopisi: Bakteri",

    # --- KANSER BELİRTEÇLERİ (TÜMÖR MARKERLARI) ---
    "PSA (Prostat Spesifik Antijen - Total)", 
    "CEA (Karsinoembriyonik Antijen)", 
    "CA 19-9", 
    "CA 125", 
    "CA 15-3", 
    "AFP (Alfa Fetoprotein)"
]

def make_id(text: str) -> str:
    tr_map = str.maketrans("ğüşıöç", "gusioc")
    text = text.lower().translate(tr_map)
    for ch in [" ", "(", ")", "%", "-", ".", "/"]:
        text = text.replace(ch, "_")
    return "_".join(filter(None, text.split("_")))

dataset = []
completed_ids = set()

if os.path.exists(FILE_NAME):
    try:
        with open(FILE_NAME, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            completed_ids = {item.get("id") for item in dataset if "id" in item}
    except Exception:
        dataset = []

missing_tests = [t for t in analysis_list if make_id(t) not in completed_ids]
processed_today = 0

def safe_json_parse(text: str):
    text = text.replace('\ufeff', '')
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("JSON parse hatası")
    return json.loads(text[start:end])

for parametre in missing_tests:
    if processed_today >= MAX_PER_RUN:
        break

    current_id = make_id(parametre)
    print(f"[{processed_today + 1}/{MAX_PER_RUN}] İşleniyor: {parametre}", end="\r")

    prompt = f"""
Return ONLY valid JSON in Turkish language. Use proper Turkish characters (ç, ğ, ı, ö, ş, ü).
{{
  "test_adi": "{parametre}",
  "kategori": "Kan / İdrar / Hormon",
  "genel_aciklama": "Tıbbi açıklama",
  "yukseklik_anlami": "En az 3 madde",
  "dusukluk_anlami": "En az 3 madde",
  "normal_deger_notu": "Referans aralığı"
}}
"""
    try:
        response = model.generate_content(prompt)
        data = safe_json_parse(response.text)
        data["id"] = current_id
        dataset.append(data)
        processed_today += 1
        
        with open(FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        time.sleep(4)
    except Exception:
        time.sleep(6)

print(f"\nBitti. {processed_today} yeni kayıt eklendi.")