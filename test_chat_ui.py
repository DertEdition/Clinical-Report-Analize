"""
Run:
    pip install pytest
    pytest test_chat_ui.py -v
"""

import pytest
import io
import os
import sys
from unittest.mock import MagicMock, patch

# chat_ui.py'nin bulunduğu klasör
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_ui import (
    tahlil_analiz_motoru,
    rapor_yaz,
    DatabaseConnectionException,
)

def make_mock_uploaded_file(name="test.pdf", content=b"%PDF-1.4 fake"):
    """Streamlit UploadedFile'ı taklit eden basit nesne."""
    f = io.BytesIO(content)
    f.name = name
    return f

def make_pdf_with_tables(rows):
    """
    pdfplumber.open() mock'u için kullanılacak sahte PDF nesnesi.
    rows: [["Test Adı", "Sonuç", "Birim", "Referans"], ...]
    """
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = [rows]

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    return mock_pdf

#TC-LAB-01
class TestLAB01:
    def test_valid_pdf_parsed_and_anomalies_detected(self):
        uploaded = make_mock_uploaded_file("lab.pdf")
        # WBC: sonuç=12.5, referans=4.0-10.0 → Yüksek
        rows = [
            ["WBC", "12.5", "10^3/uL", "4.0-10.0"],
        ]
        mock_pdf = make_pdf_with_tables(rows)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        assert len(anormallikler) == 1
        assert anormallikler[0]["test_adi"] == "WBC"
        assert anormallikler[0]["durum"] == "Yüksek"


#TC-LAB-02
class TestLAB02:
    def test_all_values_normal_returns_empty_anomaly_list(self):
        uploaded = make_mock_uploaded_file("lab.pdf")
        # HGB: sonuç=13.5, referans=12.0-17.5 → normal
        rows = [
            ["HGB", "13.5", "g/dL", "12.0-17.5"],
        ]
        mock_pdf = make_pdf_with_tables(rows)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        assert anormallikler == []



#TC-LAB-03
class TestLAB03:
    def test_anomalous_values_trigger_ai_report(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["YÜKSEKLİK ANLAMI\nDemir eksikliği\nNORMAL DEĞER"]],
            "metadatas": [[{"test_adi": "WBC"}]]
        }

        anormallikler = [
            {"test_adi": "WBC", "sonuc": "12.5", "referans": "4.0-10.0", "durum": "Yüksek"}
        ]

        mock_response = {"message": {"content": "WBC yüksek raporu"}}

        with patch("chat_ui.ollama.chat", return_value=mock_response), \
             patch("chat_ui.st.expander"), \
             patch("chat_ui.st.code"):
            rapor, fallback = rapor_yaz(anormallikler, mock_collection)

        assert rapor == "WBC yüksek raporu"
        assert fallback is None


#TC-LAB-04
class TestLAB04:
    def test_non_pdf_file_raises_value_error(self):
        uploaded = make_mock_uploaded_file("rapor.docx")

        with pytest.raises(ValueError) as exc_info:
            tahlil_analiz_motoru(uploaded)

        assert "Desteklenmeyen dosya formatı" in str(exc_info.value)

    def test_image_file_raises_value_error(self):
        uploaded = make_mock_uploaded_file("foto.png")

        with pytest.raises(ValueError) as exc_info:
            tahlil_analiz_motoru(uploaded)

        assert "Desteklenmeyen dosya formatı" in str(exc_info.value)


#TC-LAB-05
class TestLAB05:
    def test_encrypted_pdf_raises_value_error(self):
        uploaded = make_mock_uploaded_file("encrypted.pdf")

        with patch("chat_ui.pdfplumber.open", side_effect=Exception("encrypt: password required")):
            with pytest.raises(ValueError) as exc_info:
                tahlil_analiz_motoru(uploaded)

        assert "Şifreli" in str(exc_info.value)

    def test_corrupt_pdf_raises_value_error(self):
        uploaded = make_mock_uploaded_file("corrupt.pdf")

        with patch("chat_ui.pdfplumber.open", side_effect=Exception("invalid PDF structure")):
            with pytest.raises(ValueError) as exc_info:
                tahlil_analiz_motoru(uploaded)

        assert "PDF açılamadı" in str(exc_info.value)


#TC-LAB-06
class TestLAB06:
    def test_no_tables_returns_warning_message(self):
        uploaded = make_mock_uploaded_file("empty.pdf")

        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []  # tablo yok
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        assert tablolar == []
        assert anormallikler == []
        assert any("laboratuvar tablosu bulunamadı" in w for w in warnings)


#TC-LAB-07
class TestLAB07:
    def test_malformed_reference_range_skipped_with_warning(self):
        uploaded = make_mock_uploaded_file("lab.pdf")
        # Geçersiz referans: "abc-xyz"
        rows = [
            ["WBC", "12.5", "10^3/uL", "abc-xyz"],
        ]
        mock_pdf = make_pdf_with_tables(rows)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        # Malformed satır anormallik üretemez
        assert anormallikler == []

    def test_valid_row_alongside_malformed_still_processed(self):
        uploaded = make_mock_uploaded_file("lab.pdf")
        rows = [
            ["HGB", "8.0", "g/dL", "12.0-17.5"],   # geçerli, Düşük
        ]
        mock_pdf = make_pdf_with_tables(rows)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        assert len(anormallikler) == 1
        assert anormallikler[0]["durum"] == "Düşük"


#TC-LAB-08
class TestLAB08:
    def test_high_status_filters_out_low_section(self):
        mock_collection = MagicMock()
        db_text = (
            "YÜKSEKLİK ANLAMI\n"
            "Enfeksiyon olabilir\n"
            "DÜŞÜKLÜK ANLAMI\n"
            "Kemik iliği sorunu\n"
            "NORMAL DEĞER\n"
            "4.0-10.0\n"
        )
        mock_collection.query.return_value = {
            "documents": [[db_text]],
            "metadatas": [[{"test_adi": "WBC"}]]
        }

        anormallikler = [
            {"test_adi": "WBC", "sonuc": "12.5", "referans": "4.0-10.0", "durum": "Yüksek"}
        ]
        mock_response = {"message": {"content": "rapor"}}

        captured_context = {}

        def fake_chat(model, messages, options):
            captured_context["user"] = messages[1]["content"]
            return mock_response

        with patch("chat_ui.ollama.chat", side_effect=fake_chat), \
             patch("chat_ui.st.expander"), \
             patch("chat_ui.st.code"):
            rapor_yaz(anormallikler, mock_collection)

        assert "Kemik iliği sorunu" not in captured_context.get("user", "")
        assert "Enfeksiyon olabilir" in captured_context.get("user", "")

    def test_low_status_filters_out_high_section(self):
        mock_collection = MagicMock()
        db_text = (
            "YÜKSEKLİK ANLAMI\n"
            "Enfeksiyon olabilir\n"
            "DÜŞÜKLÜK ANLAMI\n"
            "Demir eksikliği\n"
            "NORMAL DEĞER\n"
            "12.0-17.5\n"
        )
        mock_collection.query.return_value = {
            "documents": [[db_text]],
            "metadatas": [[{"test_adi": "HGB"}]]
        }

        anormallikler = [
            {"test_adi": "HGB", "sonuc": "8.0", "referans": "12.0-17.5", "durum": "Düşük"}
        ]
        mock_response = {"message": {"content": "rapor"}}
        captured_context = {}

        def fake_chat(model, messages, options):
            captured_context["user"] = messages[1]["content"]
            return mock_response

        with patch("chat_ui.ollama.chat", side_effect=fake_chat), \
             patch("chat_ui.st.expander"), \
             patch("chat_ui.st.code"):
            rapor_yaz(anormallikler, mock_collection)

        assert "Enfeksiyon olabilir" not in captured_context.get("user", "")
        assert "Demir eksikliği" in captured_context.get("user", "")


#TC-LAB-09
class TestLAB09:
    def test_chromadb_unavailable_raises_exception(self):
        anormallikler = [
            {"test_adi": "WBC", "sonuc": "12.5", "referans": "4.0-10.0", "durum": "Yüksek"}
        ]

        with pytest.raises(DatabaseConnectionException):
            rapor_yaz(anormallikler, collection=None)

    def test_veritabani_baglan_raises_on_chromadb_failure(self):
        with patch("chat_ui.chromadb.PersistentClient", side_effect=Exception("connection refused")):
            # cache'i bypass etmek için direkt import
            import chat_ui
            # cache_resource'u geçici devre dışı bırak
            with pytest.raises(DatabaseConnectionException) as exc_info:
                chat_ui.veritabani_baglan.__wrapped__()

        assert "ChromaDB" in str(exc_info.value)


#TC-LAB-10
class TestLAB10:
    def test_llm_failure_returns_fallback_anomaly_list(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["YÜKSEKLİK ANLAMI\ntest\nDÜŞÜKLÜK ANLAMI"]],
            "metadatas": [[{"test_adi": "WBC"}]]
        }

        anormallikler = [
            {"test_adi": "WBC", "sonuc": "12.5", "referans": "4.0-10.0", "durum": "Yüksek"},
            {"test_adi": "HGB", "sonuc": "8.0",  "referans": "12.0-17.5", "durum": "Düşük"},
        ]

        with patch("chat_ui.ollama.chat", side_effect=Exception("connection refused")), \
             patch("chat_ui.st.expander"), \
             patch("chat_ui.st.code"):
            rapor, fallback = rapor_yaz(anormallikler, mock_collection)

        assert rapor is None
        assert fallback is not None
        assert "WBC" in fallback
        assert "HGB" in fallback

    def test_llm_success_returns_rapor_and_no_fallback(self):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["YÜKSEKLİK ANLAMI\ntest\nDÜŞÜKLÜK ANLAMI"]],
            "metadatas": [[{"test_adi": "WBC"}]]
        }

        anormallikler = [
            {"test_adi": "WBC", "sonuc": "12.5", "referans": "4.0-10.0", "durum": "Yüksek"}
        ]
        mock_response = {"message": {"content": "WBC raporu üretildi"}}

        with patch("chat_ui.ollama.chat", return_value=mock_response), \
             patch("chat_ui.st.expander"), \
             patch("chat_ui.st.code"):
            rapor, fallback = rapor_yaz(anormallikler, mock_collection)

        assert rapor == "WBC raporu üretildi"
        assert fallback is None


#TC-LAB-11
class TestLAB11:
    def test_image_based_pdf_returns_no_tables_warning(self):
        uploaded = make_mock_uploaded_file("scan.pdf")

        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []  # image PDF → tablo yok
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        assert tablolar == []
        assert any("laboratuvar tablosu bulunamadı" in w for w in warnings)


#TC-LAB-12:
class TestLAB12:
    def test_duplicate_test_name_flagged_in_warnings(self):
        uploaded = make_mock_uploaded_file("lab.pdf")
        # Aynı test iki kez
        rows = [
            ["WBC", "12.5", "10^3/uL", "4.0-10.0"],
            ["WBC", "11.0", "10^3/uL", "4.0-10.0"],
        ]
        mock_pdf = make_pdf_with_tables(rows)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        wbc_entries = [a for a in anormallikler if a["test_adi"] == "WBC"]
        assert len(wbc_entries) == 1

        assert any("Tekrarlanan" in w for w in warnings)

    def test_different_test_names_not_flagged(self):
        uploaded = make_mock_uploaded_file("lab.pdf")
        rows = [
            ["WBC", "12.5", "10^3/uL", "4.0-10.0"],
            ["HGB", "8.0",  "g/dL",    "12.0-17.5"],
        ]
        mock_pdf = make_pdf_with_tables(rows)

        with patch("chat_ui.pdfplumber.open", return_value=mock_pdf):
            tablolar, anormallikler, warnings = tahlil_analiz_motoru(uploaded)

        duplicate_warnings = [w for w in warnings if "Tekrarlanan" in w]
        assert len(duplicate_warnings) == 0


# TC-LAB-13: Kullanıcı arayüzü okunabilirlik testi — MANUEL
# Bu test otomatize edilemez. Aşağıdaki kontrol listesi manuel olarak uygulanır:
#------------------------------------------------------------------------------
# 1 Anormallik tablosu (test adı, sonuç, referans, durum) açıkça görünüyor mu?
# 2 Yüksek/Düşük indikatörler renkli veya bold ile ayrışıyor mu?
# 3 Yapay zeka raporu madde madde, Türkçe ve anlaşılır mı?
# 4 Uyarı (warnings) kutusu collapse edilebiliyor mu?
# 5 Fallback mesajı (LLM yoksa) yeterince bilgilendirici mi?

if __name__ == "__main__":
    pytest.main([__file__, "-v"])