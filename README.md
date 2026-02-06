# 🧬 Clinical Report Analyzer

## 🛠️ Kurulum Rehberi

Projeyi çalıştırmak için aşağıdaki adımları sırasıyla uygulayın.

### 1. Ön Hazırlıklar

* **Python 3.10** yüklü olmalıdır.
* **Ollama** uygulaması bilgisayarınızda kurulu ve çalışıyor olmalıdır. ([İndirmek için tıklayın](https://ollama.com))

### 2. Projeyi Klonlayın

git clone [https://github.com/kullanici_adiniz/repo_adiniz.git](https://github.com/kullanici_adiniz/repo_adiniz.git)
cd repo_adiniz 

### 3. VENV Kurulumu

python3.10 -m venv clinical_report_venv
source clinical_report_venv/bin/activate


### 4. Kütüphanelerin Yüklenmesi

pip install -r requirements.txt

### 5. Database Kurulumu

Terminalde şu komutu çalıştırarak Llama 3.2 modelini indirin:
    ollama run llama3.2
    (Model indikten sonra >>> işareti çıkınca pencereyi kapatabilirsiniz.)

Proje içinde gelen hazır tıbbi veri setini (buyuk_medikal_dataset.json) vektör veritabanına işlemek için:
    python import_dataset.py

### 6. Uygulamayı Çalıştırma 


streamlit run chat_ui.py

### (Opsiyonel) Veri Seti Üretimi

python dataset_generator.py komutu ile generative ai kullanılarak günlük prompt hakkı kadar veri üretilebilir.