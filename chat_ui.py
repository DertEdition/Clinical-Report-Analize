import streamlit as st
import os
import json
import re
import pandas as pd
import pdfplumber
from google import genai 
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
    
    def metni_akilli_filtrele(ham_metin, hasta_durumu):
        #TC-LAB-08
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
    # ---------------------------------------------------------

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

    with st.expander("🕵️ Debug: Modele Giden (Filtrelenmiş) Veri", expanded=True):
        st.code(context_data, language="text")
    # -----------------------------------------

    # 3. SYSTEM PROMPT
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
    
    # 4. USER PROMPT
    user_prompt = f"""
    Aşağıdaki temizlenmiş hasta verilerini kullanarak raporu yaz:
    
    {context_data}
    """

    #TC-LAB-10  
    try:
        response = ollama.chat(
            model=LOCAL_MODEL, 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0.1} 
        )
        return response['message']['content'], None
    except Exception as e:
        fallback = "\n".join([
            f"- **{b['test_adi']}**: {b['durum']} (Sonuç: {b['sonuc']} | Referans: {b['referans']})"
            for b in anormallikler
        ])
        return None, fallback



#-----user interface-----
st.title("🩺 Akıllı Tahlil Analiz Asistanı")
st.write("PDF'i yükleyin, sistem referans dışı değerleri bulup yorumlasın.")

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
