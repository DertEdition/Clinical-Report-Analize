import streamlit as st
import os
import json
import re
import pandas as pd
import pdfplumber
import chromadb
import ollama
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

st.set_page_config(page_title="Medikal Analiz Asistanı", layout="wide", page_icon="🧬")
load_dotenv(override=True)

LOCAL_MODEL = "llama3.2"

class DatabaseConnectionException(Exception):
    pass

#TC-LAB-09
@st.cache_resource
def veritabani_baglan():
    try:
        client = chromadb.PersistentClient(path="./medikal_db")
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        return client.get_collection("tahlil_bilgileri", embedding_function=embedding_func)
    except Exception as e:
        raise DatabaseConnectionException(f"ChromaDB bağlantısı kurulamadı: {str(e)}")
    
def tahlil_analiz_motoru(uploaded_file):
    anormallikler = []
    tum_veriler = []
    warnings = []
    eklenen_testler = set()

    YASAKLI_BIRIMLER = [
        "g/dL", "mg/dL", "ug/dL", "uL", "IU/L", "U/L", "%", "ng/mL", 
        "mm/h", "fL", "pg", "deg", "10^3/uL", "10^6/uL", "mU/L", "None", "nan",
        "Sonuç", "Referans", "Birim", "Değer"
    ]

    #TC-LAB-04
    filename = getattr(uploaded_file, "name", "")
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"Desteklenmeyen dosya formatı: '{filename}'. Yalnızca PDF dosyaları kabul edilir.")

    #TC-LAB-05
    try:
        pdf_obj = pdfplumber.open(uploaded_file)
    except Exception as e:
        err = str(e).lower()
        if "encrypt" in err or "password" in err:
            raise ValueError("Şifreli veya parola korumalı PDF işlenemiyor. Kilitsiz bir PDF yükleyin.")
        raise ValueError(f"PDF açılamadı: {str(e)}")

    with pdf_obj as pdf:
        table_found = False

        for page in pdf.pages:
            tables = page.extract_tables()

            for table in tables:
                if table:
                    table_found = True
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
                            except:
                                #TC-LAB-07
                                warnings.append(f"Referans aralığı okunamadı, satır atlandı: '{cell}'") 
                                continue

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
                        
                        if cell_text in YASAKLI_BIRIMLER: continue 
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

                        #TC-LAB-12
                        if clean_name in eklenen_testler:
                            warnings.append(f"Tekrarlanan test adı atlandı: '{test_adi}'")
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
        #TC-LAB-06
        if not table_found:
            warnings.append("PDF içinde analiz edilebilir laboratuvar tablosu bulunamadı.")                            
                            
    return tum_veriler, anormallikler, warnings 


def rapor_yaz(anormallikler, collection):
    if not collection: 
        raise DatabaseConnectionException("Vektör veritabanına erişilemiyor.")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def metni_akilli_filtrele(ham_metin, hasta_durumu):
        #TC-LAB-08
        if "Yüksek" in hasta_durumu:
            aranan_anahtar = "yukseklik_anlami"
        elif "Düşük" in hasta_durumu:
            aranan_anahtar = "dusukluk_anlami"
        else:
            return ham_metin

        try:
            veri = json.loads(ham_metin)
            icerik = veri.get(aranan_anahtar, "")

            if isinstance(icerik, list):
                return "\n".join(f"- {madde}" for madde in icerik if str(madde).strip())
            return str(icerik).strip() if icerik else ham_metin

        except (json.JSONDecodeError, TypeError):
            pass

        satirlar = ham_metin.split('\n')
        filtrelenmis = ""

        if "Yüksek" in hasta_durumu:
            baslangic = "YÜKSEKLİK ANLAMI"
            bitis    = "DÜŞÜKLÜK ANLAMI"
        else:
            baslangic = "DÜŞÜKLÜK ANLAMI"
            bitis    = "NORMAL DEĞER"

        kayit = False
        for satir in satirlar:
            if baslangic in satir.upper():
                kayit = True
                continue
            if bitis in satir.upper():
                kayit = False
                break
            if kayit:
                filtrelenmis += satir + "\n"

        return filtrelenmis.strip() if len(filtrelenmis.strip()) > 10 else ham_metin

    # ---------------------------------------------------------

    SYSTEM_PROMPT = """
    Sen uzman bir Türk tıbbi asistanısın. Sana TEK bir laboratuvar testi veriliyor.
    Sadece o test için aşağıdaki formatı uygula, başka hiçbir şey yazma:

    **[Test Adı]**
    **Sonuç:** [Değer] | **Referans:** [Aralık] | **Durum:** Değeriniz referans aralığının [üzerindedir / altındadır].

    **Olası Sebepler:**
    - [Sebep 1]
    - [Sebep 2]
    - [Sebep 3]

    **En Yaygın Neden:** [Bu durumun günlük hayatta en sık görülen nedeni, 1-2 cümle.]

    **Günlük Hayat Önerileri:**
    - [Beslenme ile ilgili somut öneri]
    - [Aktivite / yaşam tarzı önerisi]
    - [Varsa kaçınılması gereken şey]

    **Doktora Başvurma Zamanı:** [Bu değerin hangi belirtilerle birlikte görülmesi durumunda doktora gidilmesi gerektiğini 1-2 cümleyle açıkla.]

    !!! ZORUNLU KURALLAR — İHLAL ETMEDİĞİNİ KONTROL ET !!!
    1. Yanıtın TAMAMEN Türkçe olmalı. Tek bir İngilizce kelime bile yasak.
    2. Şu İngilizce kelimeleri ASLA kullanma, Türkçe karşılıklarını yaz:
    - "balanced"           → "dengeli"
    - "regular/regularly"  → "düzenli/düzenli olarak"
    - "detailed"           → "ayrıntılı"
    - "products"           → "ürünler"
    - "diet"               → "beslenme düzeni"
    - "analysis"           → "analiz"
    - "levels"             → "seviyeleri"
    - "symptoms"           → "belirtiler"
    - "treatment"          → "tedavi"
    - "check"              → "kontrol"
    - "chronic"            → "kronik"
    - "deficiency"         → "eksikliği"
    - "infection"          → "enfeksiyon"
    - "inflammation"       → "iltihaplanma"
    - "associate"          → "ilgi"
   
    3. "SERUM", "PLAZMA" kelimelerini başlık veya etiket olarak yazma.
    4. Yanıtı yazmadan önce İngilizce kelime kullanıp kullanmadığını zihninden kontrol et.
    """

    INGILIZCE_SOZLUK = {
        r'\bbalanced\b':       'dengeli',
        r'\bregularly\b':      'düzenli olarak',
        r'\bregular\b':        'düzenli',
        r'\bdetailed\b':       'ayrıntılı',
        r'\bproducts?\b':      'ürünler',
        r'\bdiet\b':           'beslenme düzeni',
        r'\banalysis\b':       'analiz',
        r'\blevels?\b':        'seviyeleri',
        r'\bsymptoms?\b':      'belirtiler',
        r'\btreatment\b':      'tedavi',
        r'\bdoctor\b':         'doktor',
        r'\bcheck\b':          'kontrol',
        r'\bhealth\b':         'sağlık',
        r'\bblood\b':          'kan',
        r'\btest\b':           'test',
        r'\brisk\b':           'risk',
        r'\bchronic\b':        'kronik',
        r'\bdeficiency\b':     'eksikliği',
        r'\binfection\b':      'enfeksiyon',
        r'\binflammation\b':   'iltihaplanma',
        r'\ associate\b':       'ilgi'
    }

    def ingilizce_temizle(metin):
        for pattern, turkce in INGILIZCE_SOZLUK.items():
            metin = re.sub(pattern, turkce, metin, flags=re.IGNORECASE)
        return metin


    def tek_test_isle(bulgu):
        try:
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
            else:
                ozel_bilgi = "Veritabanında bu test için bilgi bulunamadı."
        except Exception:
            ozel_bilgi = "Veritabanı sorgusu başarısız."

        user_prompt = f"""
        TEST: {bulgu['test_adi']}
        DURUM: {bulgu['durum']} (Sonuç: {bulgu['sonuc']} | Referans: {bulgu['referans']})

        SEBEPLER (veritabanından):
        {ozel_bilgi}
        """

        try:
            response = ollama.chat(
                model=LOCAL_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': user_prompt},
                ],
                options={
                    'temperature': 0.1,
                    'num_ctx': 1024,
                    'num_predict': 512,
                    'keep_alive': '10m',
                }
            )
            temiz_rapor = ingilizce_temizle(response['message']['content'])
            return bulgu['test_adi'], temiz_rapor, None
        except Exception as e:
            fallback = (
                f"- **{bulgu['test_adi']}**: {bulgu['durum']} "
                f"(Sonuç: {bulgu['sonuc']} | Referans: {bulgu['referans']})"
            )
            return bulgu['test_adi'], None, fallback

    MAX_WORKERS = min(4, len(anormallikler))

    sonuclar = {}
    fallback_listesi = []
    debug_context = ""

    progress = st.progress(0, text="Testler analiz ediliyor...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(tek_test_isle, b): b['test_adi'] for b in anormallikler}
        tamamlanan = 0

        for future in as_completed(future_map):
            test_adi_tamamlandi = future_map[future]
            tamamlanan += 1
            progress.progress(
                tamamlanan / len(anormallikler),
                text=f"✅ {test_adi_tamamlandi} tamamlandı ({tamamlanan}/{len(anormallikler)})"
            )
            try:
                test_adi, rapor_metni, fb = future.result()
                sonuclar[test_adi] = rapor_metni
                if fb:
                    fallback_listesi.append(fb)
                debug_context += f"[{test_adi}]\n{rapor_metni or fb}\n\n"
            except Exception as exc:
                fallback_listesi.append(f"- **{test_adi_tamamlandi}**: işlem hatası ({exc})")

    progress.empty()

    #with st.expander("🕵️ Debug: Model Çıktıları", expanded=False):
    #    st.code(debug_context, language="text")

    basarili_raporlar = [
        sonuclar[b['test_adi']]
        for b in anormallikler
        if sonuclar.get(b['test_adi'])
    ]

    if basarili_raporlar:
        return "\n\n---\n\n".join(basarili_raporlar), None
    else:
        return None, "\n".join(fallback_listesi)


#-----user interface-----
st.title("🩺 Akıllı Tahlil Analiz Asistanı")
st.write("PDF'i yükleyin, sistem referans dışı değerleri bulup yorumlasın.")

@st.cache_resource
def model_on_yukle():
    try:
        ollama.chat(
            model=LOCAL_MODEL,
            messages=[{'role': 'user', 'content': 'merhaba'}],
            options={'num_predict': 1, 'keep_alive': '10m'}
        )
    except Exception:
        pass

model_on_yukle()

uploaded = st.file_uploader("PDF Yükle", type="pdf")

if uploaded:

    #TC-LAB-09
    try:
        collection = veritabani_baglan()
    except DatabaseConnectionException as e:
        st.error(f"🔴 Veritabanı Hatası (503): {e}\n\n'import_dataset.py' çalıştırarak veritabanını oluşturun.")
        st.stop()

    #TC-LAB-04/05/06/07/12
    with st.spinner("Analiz yapılıyor..."):
        try:
            tablolar, sorunlar, warnings = tahlil_analiz_motoru(uploaded)
        except ValueError as e:
            st.error(f"🔴 Dosya Hatası (400): {e}")
            st.stop()

    #TC-LAB-07/12
    if warnings:
        with st.expander(f"⚠️ Uyarılar ({len(warnings)} adet)", expanded=False):
            for w in warnings:
                st.warning(w)

    #TC-LAB-06
    if not tablolar:
        st.info("ℹ️ PDF içinde analiz edilebilir laboratuvar tablosu bulunamadı.")
        st.stop()

    with st.expander(f"📄 Okunan Tablolar ({len(tablolar)} adet)"):
        for df in tablolar:
            st.dataframe(df)

    if not sorunlar:
        st.success("✅ Tüm değerler referans aralığında.")
    else:
        st.error(f"⚠️ {len(sorunlar)} adet referans dışı değer bulundu.")
        st.table(sorunlar)

        with st.spinner("Yapay zeka yorumluyor..."):
            rapor, fallback = rapor_yaz(sorunlar, collection)

        #TC-LAB-10
        if rapor:
            st.markdown("### 📋 Yapay Zeka Raporu")
            st.markdown(rapor)
        else:
            st.warning("⚠️ Yapay zeka modeli yanıt vermedi (504). Tespit edilen anormallikler:")
            st.markdown(fallback)