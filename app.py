import streamlit as st
import joblib
import re
import time
import plotly.graph_objects as go
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =============================================================================
# 1. KONFIGURASI GLOBAL & INJEKSI CSS TINGKAT LANJUT
# =============================================================================

st.set_page_config(
    page_title="Deteksi Berita Hoax | Diskominfo Jabar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Kustom untuk estetika yang disempurnakan
def inject_custom_css():
    st.markdown("""
        <style>
            /* Tema dasar dan font */
            body {
                font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            }

            /* Efek bayangan untuk kontainer utama */
            .st-emotion-cache-1r4qj8v {
                box-shadow: 0 8px 24px rgba(0,0,0,0.08);
                border: 1px solid rgba(0,0,0,0.05);
                border-radius: 12px;
            }
            
            /* Tombol utama yang lebih profesional */
            .stButton>button {
                border-radius: 8px;
                font-weight: 600;
            }

            /* Header hasil dengan desain baru - INI TETAP BERWARNA */
            .result-header-valid, .result-header-hoax {
                padding: 1rem 1.5rem;
                border-radius: 11px 11px 0 0;
                font-weight: 700;
                font-size: 1.4rem;
                border-bottom: 2px solid;
            }
            .result-header-valid {
                background-color: #F0FFF4;
                color: #2F855A;
                border-bottom-color: #9AE6B4;
            }
            .result-header-hoax {
                background-color: #FFF5F5;
                color: #C53030;
                border-bottom-color: #FEB2B2;
            }
            
            /* Styling untuk wawasan tambahan */
            .insight-list {
                list-style-type: none;
                padding-left: 0;
                margin-top: 10px;
            }
            .insight-list li {
                margin-bottom: 12px;
                font-size: 0.95rem;
                padding-left: 1.5rem;
                position: relative;
            }
            .insight-list li::before {
                content: '■';
                position: absolute;
                left: 0;
                font-size: 0.8rem;
                top: 5px;
            }
            .insight-category {
                font-weight: 600;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
            }
            
            /* Grid untuk metrik yang sejajar sempurna di panel utama*/
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1rem;
                text-align: center;
                margin-top: 1.5rem;
                padding: 0 1rem;
                padding-bottom: 1rem; /* MENAMBAHKAN PADDING BAWAH */
            }
            
            .metric-item .label {
                font-size: 0.85rem;
                opacity: 0.7;
            }
            .metric-item .value {
                font-size: 1.1rem;
                font-weight: 600;
            }
            
            /* --- DESAIN BARU UNTUK SIDEBAR --- */
            .sidebar-header {
                text-align: center;
                padding: 0.5rem 0;
                margin-bottom: 1rem;
            }
            .sidebar-main-title {
                font-size: 2.2rem; /* Sedikit dikecilkan agar pas */
                font-weight: 700;
                letter-spacing: 1px; /* Mengurangi letter-spacing */
            }
            .sidebar-tagline {
                font-size: 0.9rem;
                opacity: 0.8;
                margin-top: -10px;
            }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Inisialisasi session state
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# =============================================================================
# 2. FUNGSI INTI & PEMUATAN SUMBER DAYA
# =============================================================================

@st.cache_resource
def load_resources():
    try:
        resources = {
            "model": joblib.load('svm_model.pkl'),
            "vectorizer": joblib.load('tfidf_vectorizer.pkl'),
            "stemmer": StemmerFactory().create_stemmer(),
            "stopwords": set(StopWordRemoverFactory().get_stop_words())
        }
        return resources
    except FileNotFoundError:
        st.error("Error: Pastikan file model 'svm_model.pkl' dan 'tfidf_vectorizer.pkl' ada di direktori yang sama.")
        return None

def run_text_preprocessing(text: str, _stemmer, _stopwords) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+|[^\w\s]', ' ', text)
    tokens = [word for word in text.split() if word not in _stopwords and len(word) > 1]
    tokens = [_stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def perform_analysis(text: str, resources: dict):
    start_time = time.time()
    
    # Analisis mentah dilakukan pada teks asli sebelum pembersihan
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.split()) > 3]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    raw_analysis = {
        "word_count": len(text.split()),
        "avg_sentence_length": avg_sentence_length,
        "num_count": len(re.findall(r'\d', text)),
        "upper_count": len(re.findall(r'\b[A-Z]{2,}\b', text)),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "quote_count": text.count('"') + text.count("'"),
        "first_person_count": len(re.findall(r'\b(saya|kami|penulis)\b', text, re.IGNORECASE)),
        "has_source": bool(re.search(r'(dilansir|menurut|sumber|dikutip|melansir)', text, re.IGNORECASE)),
        "call_to_action": len(re.findall(r'\b(bagikan|sebarkan|viralkan|share)\b', text, re.IGNORECASE)),
        "emotional_words": len(re.findall(r'(waspada|sebarkan|penting|bahaya|terungkap|fakta|bukti|viral|heboh|menggemparkan|skandal)', text, re.IGNORECASE)),
        "clickbait_phrases": len(re.findall(r'(mengejutkan|terbongkar|jangan kaget|wajib tahu|ternyata|begini|tak disangka)', text, re.IGNORECASE)),
        "entity_mentions": len(re.findall(r'\b(presiden|menteri|gubernur|polri|kpk|dpr|pemprov|pemkab|kemenkeu|bi|istana)\b', text, re.IGNORECASE)),
        "date_mentions": len(re.findall(r'\b(\d{1,2}\s(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s\d{4})\b|\b(\d{4})\b', text, re.IGNORECASE)),
    }
    
    cleaned_text = run_text_preprocessing(text, resources["stemmer"], resources["stopwords"])
    
    # Validasi setelah pembersihan
    if len(cleaned_text.split()) < 5:
        return {"error": "Teks terlalu singkat atau tidak mengandung informasi yang cukup untuk dianalisis."}

    vectorized_text = resources["vectorizer"].transform([cleaned_text])
    prediction = resources["model"].predict(vectorized_text)[0]
    probability = resources["model"].predict_proba(vectorized_text)[0]
    processing_time = time.time() - start_time
    
    return {
        "prediction": prediction,
        "confidence": probability.max(),
        "processing_time": processing_time,
        "raw_analysis": raw_analysis
    }

def create_gauge_chart(value):
    primary_color = st.get_option("theme.primaryColor") or "#0068c9"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Tingkat Keyakinan"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': primary_color},
        }))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def generate_advanced_insights(result):
    insights = {
        "Gaya Penulisan": [],
        "Struktur & Kredibilitas": [],
        "Potensi Manipulasi": []
    }
    raw = result.get('raw_analysis', {})
    is_hoax = result.get("prediction") == 1
    
    # --- Kategori 1: Gaya Penulisan ---
    if is_hoax:
        insights["Gaya Penulisan"].append(('Gaya Bahasa', 'Cenderung emotif dan mendesak, sering ditemukan dalam disinformasi.'))
    else:
        insights["Gaya Penulisan"].append(('Gaya Bahasa', 'Cenderung faktual dan objektif, ciri khas berita kredibel.'))

    if raw.get("num_count", 0) > 2:
        insights["Gaya Penulisan"].append(('Orientasi Teks', 'Berbasis data, mengandung angka yang bisa diverifikasi.'))
    else:
        insights["Gaya Penulisan"].append(('Orientasi Teks', 'Berbasis narasi, lebih fokus pada cerita yang bisa subjektif.'))

    avg_len = raw.get('avg_sentence_length', 0)
    if avg_len > 25:
        insights["Gaya Penulisan"].append(('Kompleksitas Kalimat', 'Kalimat cenderung panjang dan kompleks, mungkin sulit dipahami oleh masyarakat umum.'))
    elif avg_len < 10 and avg_len > 0:
        insights["Gaya Penulisan"].append(('Kompleksitas Kalimat', 'Kalimat sangat singkat, yang bisa jadi terlalu menyederhanakan masalah.'))
    else:
         insights["Gaya Penulisan"].append(('Kompleksitas Kalimat', 'Struktur kalimat memiliki kompleksitas yang wajar untuk sebuah berita.'))
    
    if raw.get('question_count', 0) > 1:
        insights["Gaya Penulisan"].append(('Penggunaan Pertanyaan', f'Teks ini menggunakan {raw.get("question_count", 0)} kalimat tanya. Ini bisa menjadi teknik retoris untuk menarik pembaca.'))

    # --- Kategori 2: Struktur & Kredibilitas ---
    if raw.get('has_source', False):
        insights["Struktur & Kredibilitas"].append(('Penyebutan Sumber', 'Positif. Teks menyebutkan kata kunci sumber, meningkatkan potensi kredibilitas.'))
    else:
        insights["Struktur & Kredibilitas"].append(('Penyebutan Sumber', 'Negatif. Tidak ada penyebutan sumber eksplisit, perlu dipertanyakan.'))

    if raw.get('quote_count', 0) > 1:
        insights["Struktur & Kredibilitas"].append(('Kutipan Langsung', f'Positif. Ditemukan {raw.get("quote_count", 0)} tanda kutip, mengindikasikan adanya kutipan dari narasumber.'))
    
    if raw.get('entity_mentions', 0) > 0:
        insights["Struktur & Kredibilitas"].append(('Penyebutan Entitas', f'Positif. Terdeteksi {raw.get("entity_mentions", 0)} penyebutan institusi atau jabatan resmi.'))
    else:
        insights["Struktur & Kredibilitas"].append(('Penyebutan Entitas', 'Tidak ada penyebutan entitas resmi. Informasi yang tidak menyebutkan siapa, apa, dan di mana cenderung kurang kredibel.'))

    if raw.get('date_mentions', 0) > 0:
        insights["Struktur & Kredibilitas"].append(('Konteks Waktu', 'Positif. Adanya penyebutan tanggal atau tahun menempatkan informasi dalam kerangka waktu yang jelas.'))

    word_count = raw.get('word_count', 0)
    if word_count < 100:
        insights["Struktur & Kredibilitas"].append(('Kedalaman Konten', f'Cukup singkat ({word_count} kata), kemungkinan tidak memberikan konteks yang mendalam.'))
    else:
        insights["Struktur & Kredibilitas"].append(('Kedalaman Konten', f'Panjang artikel wajar ({word_count} kata), memungkinkan adanya pembahasan yang cukup.'))

    # --- Kategori 3: Potensi Manipulasi ---
    emotional_words_count = raw.get('emotional_words', 0)
    if emotional_words_count > 1:
        insights["Potensi Manipulasi"].append(('Kata Emotif', f"Ditemukan {emotional_words_count} kata kunci pemancing emosi (contoh: 'sebarkan', 'bahaya'). Ini adalah Red Flag."))
    
    clickbait_count = raw.get('clickbait_phrases', 0)
    if clickbait_count > 0:
        insights["Potensi Manipulasi"].append(('Frasa Umpan Klik', f"Terdeteksi {clickbait_count} frasa yang umum digunakan sebagai umpan klik (clickbait)."))

    exclamation_count = raw.get('exclamation_count', 0)
    if exclamation_count > 2:
        insights["Potensi Manipulasi"].append(('Sensasionalisme', f"Penggunaan {exclamation_count} tanda seru menunjukkan gaya penulisan yang berpotensi sensasional."))

    upper_count = raw.get('upper_count', 0)
    if upper_count > 2:
        insights["Potensi Manipulasi"].append(('Penekanan Berlebih', f'Ditemukan {upper_count} kata dengan huruf kapital berlebih. Ini sering digunakan untuk menciptakan kesan urgensi.'))

    if raw.get('first_person_count', 0) > 0:
        insights["Potensi Manipulasi"].append(('Subjektivitas', 'Penggunaan kata ganti orang pertama ("saya", "kami") terdeteksi, mengindikasikan tulisan bersifat opini, bukan berita objektif.'))

    if raw.get('call_to_action', 0) > 0:
        insights["Potensi Manipulasi"].append(('Ajakan Bertindak', 'Red Flag. Teks ini secara eksplisit meminta untuk disebarkan. Ini adalah ciri khas utama dari disinformasi.'))

    # --- Insight Kombinasi ---
    if is_hoax and emotional_words_count > 1 and not raw.get('has_source', False):
        insights["Potensi Manipulasi"].append(('Pola Disinformasi Klasik', 'Kombinasi bahasa emotif dan ketiadaan sumber adalah pola paling umum dari berita hoaks.'))
    elif not is_hoax and raw.get('has_source', False) and raw.get('entity_mentions', 0) > 0:
        insights["Struktur & Kredibilitas"].append(('Pola Berita Kredibel', 'Kombinasi penyebutan sumber, entitas resmi, dan gaya bahasa netral adalah indikator kuat berita terpercaya.'))

    return insights

# =============================================================================
# 3. RENDER KOMPONEN UI
# =============================================================================

def render_sidebar(resources):
    with st.sidebar:
        # Header Sidebar yang Didesain Ulang
        st.markdown("""
            <div class='sidebar-header'>
                <p class='sidebar-main-title'>DETEKSI BERITA HOAX</p>
                <p class='sidebar-tagline'>Proyek Magang Diskominfo Jabar</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            st.subheader("Statistik Model")
            col1, col2 = st.columns(2)
            col1.metric("Akurasi", "98.20%", help="Akurasi model pada data uji.")
            col2.metric("Metode", "SVM", help="Support Vector Machine (SVM) digunakan sebagai model klasifikasi.")
        
        with st.container(border=True):
            st.subheader("Analisis Cepat")
            contoh_valid = "Menteri Keuangan Sri Mulyani Indrawati menyatakan bahwa realisasi sementara Anggaran Pendapatan dan Belanja Negara (APBN) 2024 mencatatkan kinerja positif hingga akhir Mei. Pendapatan negara mencapai Rp1.123,5 triliun atau 40,1 persen dari target, sementara belanja negara terealisasi sebesar Rp1.144,7 triliun atau 34,4 persen."
            contoh_hoax = "SEBARKAN!! Beredar kabar bahwa minuman bersoda dapat menyembuhkan penyakit COVID-19 dalam waktu singkat setelah dikonsumsi secara rutin. Informasi ini menyebar cepat di grup WhatsApp dan media sosial, mengklaim bahwa kandungan soda mampu membunuh virus di tenggorokan. Faktanya, klaim tersebut tidak memiliki dasar ilmiah dan telah dibantah oleh para ahli kesehatan."
            
            b_col1, b_col2 = st.columns(2, gap="small")
            with b_col1:
                if st.button("Contoh VALID", use_container_width=True):
                    st.session_state.text_input = contoh_valid
                    st.session_state.last_result = None
            with b_col2:
                if st.button("Contoh HOAX", use_container_width=True):
                    st.session_state.text_input = contoh_hoax
                    st.session_state.last_result = None

        with st.container(border=True):
            st.caption("Disclaimer: Hasil prediksi adalah indikasi, bukan kebenaran absolut. Selalu lakukan verifikasi silang dari sumber terpercaya.")


def render_main_panel(resources):
    st.title("Platform Deteksi Berita Hoax")
    st.markdown("Selamat datang di Platform Deteksi Berita Hoax. Masukkan teks berita untuk dianalisis apakah terindikasi sebagai hoaks atau fakta berdasarkan pola linguistik.")
    st.divider()

    # Tempat untuk menampilkan peringatan
    alert_placeholder = st.empty()

    with st.form(key='analysis_form'):
        st.text_area(
            "Tempelkan konten berita lengkap di sini untuk dianalisis:",
            key='text_input', height=250,
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button(label="Periksa Berita", use_container_width=True)

    if submit_button:
        if st.session_state.text_input.strip():
            with st.spinner("Melakukan analisis mendalam..."):
                time.sleep(0.5)
                result = perform_analysis(st.session_state.text_input, resources)
                if "error" in result:
                    alert_placeholder.warning(result["error"])
                    st.session_state.last_result = None
                else:
                    st.session_state.last_result = result
        else:
            alert_placeholder.warning("Harap masukkan teks berita terlebih dahulu untuk dianalisis.")
            st.session_state.last_result = None
    
    if st.session_state.last_result:
        render_results_card(st.session_state.last_result)

def render_results_card(result):
    is_hoax = result["prediction"] == 1
    confidence = result["confidence"] * 100
    
    with st.container(border=True):
        header_class = "result-header-hoax" if is_hoax else "result-header-valid"
        header_text = "Terindikasi HOAX" if is_hoax else "Terindikasi VALID"
        st.markdown(f"<div class='{header_class}'>{header_text}</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(create_gauge_chart(confidence), use_container_width=True)
            
            status_text = "Hoax" if is_hoax else "Valid"
            if confidence > 95: interp_text = "Sangat Yakin"
            elif confidence > 80: interp_text = "Cukup Yakin"
            else: interp_text = "Perlu Verifikasi"
            time_text = f"{result['processing_time']:.2f}"
            
            st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="label">Status</div>
                        <div class="value">{status_text}</div>
                    </div>
                    <div class="metric-item">
                        <div class="label">Interpretasi</div>
                        <div class="value">{interp_text}</div>
                    </div>
                    <div class="metric-item">
                        <div class="label">Waktu Analisis</div>
                        <div class="value">{time_text} s</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("Wawasan & Karakteristik Teks")
            st.markdown("Analisis ini mengidentifikasi beberapa karakteristik dari teks yang Anda masukkan:")
            
            categorized_insights = generate_advanced_insights(result)
            
            for category, insights in categorized_insights.items():
                if insights:
                    st.markdown(f"<p class='insight-category'>{category}</p>", unsafe_allow_html=True)
                    insight_html = "<ul class='insight-list'>"
                    for title, text in insights:
                        insight_html += f"<li><div><strong>{title}:</strong> {text}</div></li>"
                    insight_html += "</ul>"
                    st.markdown(insight_html, unsafe_allow_html=True)

# =============================================================================
# 4. EKSEKUSI APLIKASI
# =============================================================================

if __name__ == "__main__":
    resources = load_resources()
    if resources:
        render_sidebar(resources)
        render_main_panel(resources)

