# -*- coding: utf-8 -*-
"""
Purwarupa Early Warning System — Prediksi Kategori IPK Berbasis Faktor Eksternal
Standar pelaporan: TRIPOD (Transparent Reporting) & NIST AI Risk Management Framework
Mockup demo sidang MBKM Penelitian — Rayhandi Zulmi (00000103940)
Model: FNN vs BiLSTM + Attention (Sekuens Tematik 8 Domain)

Jalankan:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------- konfigurasi
st.set_page_config(
    page_title="EWS Prediksi IPK — FNN & BiLSTM+Attention",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA = Path(__file__).parent / "data"
NAVY, BLUE, AMBER, RED, GREEN = "#1A2332", "#1C5D8C", "#E8A33D", "#C0392B", "#2E7D32"
TEAL, PURPLE, GRAY = "#0E9594", "#7B5EA7", "#95A5A6"

# ⚠️ GANTI dengan DOMAIN_ORDER asli dari notebook-mu sebelum sidang.
DOMAIN_ORDER = [
    "Domain 1 - Sosial-Ekonomi Keluarga",
    "Domain 2 - Dukungan Akademik Keluarga",
    "Domain 3 - Tempat Tinggal & Lingkungan Fisik",
    "Domain 4 - Fasilitas Belajar & Infrastruktur",
    "Domain 5 - Status Kerja & Beban Aktivitas",
    "Domain 6 - Gaya Hidup & Hobi",
    "Domain 7 - Kesejahteraan Psikososial",
    "Domain 8 - Konteks Temporal (Angkatan)",
]


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown(f"<h2 style='color:{BLUE};'>🎓 EWS PREDIKSI IPK</h2>", unsafe_allow_html=True)
st.sidebar.caption(
    "**Purwarupa Early Warning System** berbasis faktor eksternal.\n\n"
    "Komparasi **FNN** vs **BiLSTM + Attention (Sekuens Tematik)**. "
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Alur:",
    [
        "1. Ringkasan Penelitian",
        "2. Penanganan Class Imbalance",
        "3. Komparasi FNN vs BiLSTM",
        "4. Faktor Penggerak Eksternal",
        "5. Simulasi Prediksi (Demo)",
        "6. Tata Kelola Risiko AI",
    ],
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Peneliti:** Rayhandi Zulmi\n\n"
    "**NIM:** 00000103940\n\n"
    "Program Studi Sistem Informasi - **UMN 2026**\n\n"
    "MBKM Penelitian · LPPM-UMN"
)

# ================================================================ 1. RINGKASAN
if page.startswith("1"):
    st.title("Pengembangan & Validasi Internal Model Prediksi Kategori IPK Mahasiswa")
    st.markdown(
        "Membangun dan mengaudit model **FNN** serta **BiLSTM + Attention (Sekuens Tematik)** "
        "untuk mendeteksi dini mahasiswa berisiko akademik **semata-mata dari 40 faktor eksternal "
        "di luar kampus** dan tanpa satu pun proksi akademik internal (nilai kuis, log LMS, IPK semester lalu)."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responden Survei", "169", "Total entri primer")
    c2.metric("Sampel Pasca-Pruning", "136", "80,5% retensi responden")
    c3.metric("Fitur Eksternal", "40 Atribut", "8 domain kehidupan")
    c4.metric("Skema Validasi", "RSKF 5×3", "15 fold + hold-out 28")

    st.divider()

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Kesenjangan Metodologis yang Diserang")
        st.markdown(
            "1. **Intervensi kampus masih reaktif:** Mahasiswa baru terdeteksi setelah IPK anjlok "
            "atau SP terbit di akhir semester.\n"
            "2. **Akar masalah ada di luar kelas:** Tekanan finansial keluarga, kerja paruh waktu, "
            "kualitas tempat tinggal, dan kesejahteraan psikososial jarang masuk sistem monitoring "
            "akademik mana pun, padahal justru di situ sinyal paling awal muncul.\n"
            "3. **Distribusi IPK bersifat *top-heavy*:** Mayoritas responden berkumpul di pita nilai atas, "
            "sehingga model naif bisa terlihat \"akurat\" hanya dengan menebak kelas mayoritas terus-menerus "
            "(*majority-class bias*)."
        )

        st.subheader("Tiga Batas Solusi yang Diusulkan")
        st.markdown(
            "1. **Isolasi mutlak faktor eksternal:** Seluruh proksi internal kampus sengaja dihapus. "
            "Tujuannya bukan mengejar akurasi setinggi-tingginya, melainkan **membuktikan seberapa besar "
            "daya determinasi murni faktor luar kampus** terhadap capaian akademik.\n"
            "2. ***Pruned Binary Classification*:** Zona ambigu IPK **3,50–3,67** (33 responden) dibuang "
            "agar batas kelas *Non-CumLaude* vs *CumLaude* menjadi tegas dan *label noise* di perbatasan hilang.\n"
            "3. ***Governance Cross-Validation Engine*:** Setiap fold tidak hanya menghasilkan metrik, "
            "tapi juga **kalibrasi probabilitas, threshold optimal, permutation importance, dan peta entropi** "
            "- pondasi koridor *Safe to Automate* vs *Human-in-the-Loop*."
        )

    with right:
        st.subheader("Distribusi Kelas Pasca-Pruning (n = 136)")
        df_cls = pd.DataFrame(
            {"Kelas": ["Kelas 0 · Non-CumLaude (< 3,50)", "Kelas 1 · CumLaude (≥ 3,67)"],
             "Jumlah": [54, 82]}
        )
        fig = px.bar(df_cls, x="Kelas", y="Jumlah", text="Jumlah",
                     color="Kelas", color_discrete_sequence=[RED, GREEN])
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=10),
                          xaxis_title="", yaxis_title="Jumlah sampel")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Rasio akhir **39,7% : 60,3%** memang belum ekstrem, tetapi cukup timpang "
            "untuk membuat model \"malas\" menebak sehingga hasil dari *CumLaude* terus akan menghasilkan akurasi 60,3% "
            "tanpa model belajar apa pun. Angka **60,3% inilah yang menjadi garis dasar (*baseline naif*)** yang wajib "
            "dilampaui setiap arsitektur."
        )

    st.divider()
    st.subheader("Rekayasa Input: Apa Itu Sekuens Tematik?")
    st.markdown(
        "LSTM lahir untuk data berurutan, sementara data survei bersifat tabular. "
        "**Sekuens Tematik** adalah jembatannya: 40 fitur eksternal tidak disodorkan sebagai satu vektor datar, "
        "melainkan **ditata ulang menjadi 8 *timestep*, satu *timestep* per domain kehidupan**, lalu dipadding "
        "seragam menjadi tensor input berdimensi **8 × 20**. Dengan begitu gerbang memori LSTM dan lapisan "
        "*attention* punya sesuatu untuk dibaca berurutan: **konteks antar-domain**, bukan sekadar deretan kolom."
    )
    dcol1, dcol2 = st.columns([1, 1])
    with dcol1:
        st.dataframe(
            pd.DataFrame({"Timestep": [f"t{i+1}" for i in range(8)], "Domain Kehidupan": DOMAIN_ORDER}),
            use_container_width=True, hide_index=True,
        )
    with dcol2:
        st.success(
            "**Kenapa ini bukan sekadar kosmetik?** tanpa penataan tematik, LSTM standar pada data tabular "
            "praktis tidak punya keunggulan struktural apa pun dibanding FNN, dan itu persis yang terjadi "
            "pada eksperimen awal. Setelah sekuens tematik diterapkan, **BiLSTM + Attention** memperoleh "
            "sesuatu yang bisa ia manfaatkan yaitu dengan membaca 8 domain **dua arah** dan memberi **bobot adaptif** "
            "pada domain yang paling informatif untuk tiap mahasiswa.\n\n"
            "Bukti kuantitatifnya ada di halaman **3 · Komparasi**."
        )

# ================================================================ 2. IMBALANCE
elif page.startswith("2"):
    st.title("Penanganan Class Imbalance — Tiga Lapis Pertahanan")
    st.caption(
        "Distribusi IPK yang top-heavy membuat kelas minoritas (Non-CumLaude) mudah terabaikan model. "
        "Tiga lapis penanganan diterapkan berurutan: sebelum, saat, dan sesudah pelatihan."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rasio kelas 0 : 1", "54 : 82", "≈ 1 : 1,52")
    m2.metric("Kelas minoritas", "39,7%", "Non-CumLaude (< 3,50)")
    m3.metric("Bobot Kelas 0", "1,26", "penalti error diperbesar")
    m4.metric("Bobot Kelas 1", "0,83", "penalti error diperkecil")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lapis 1 — Pruned Binary Classification (Pra-pelatihan)")
        fig = go.Figure()
        for nama, jml, warna in [
            ("Kelas 0 · Non-CumLaude", 54, RED),
            ("Zona ambigu 3,50–3,67 dibuang", 33, "#B0B7BF"),
            ("Kelas 1 · CumLaude", 82, GREEN),
        ]:
            fig.add_bar(y=["169 responden"], x=[jml], name=nama, orientation="h",
                        marker_color=warna, text=[jml], textposition="inside")
        fig.update_layout(barmode="stack", height=220, margin=dict(t=20, b=10),
                          legend=dict(orientation="h", y=-0.35),
                          xaxis_title="Jumlah responden", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Mekanisme.** 33 responden di pita IPK **3,50–3,67** dibuang; 169 responden → **136 sampel**.\n\n"
            "**Justifikasi metodologis.** Responden di zona ini secara substansi tidak berbeda satu sama lain, "
            "tetapi label biner memaksa mereka dipisah ke dua kelas berlawanan. Akibatnya model dilatih untuk "
            "membedakan hal yang memang tidak bisa dibedakan, maka inilah *label noise* di perbatasan. "
            "Membuangnya menghasilkan **margin kelas yang bersih** (*clean-margin reformulation*).\n\n"
            "Angka akurasi jadi naik sebagian karena "
            "kasus-kasus tersulit sudah dikeluarkan. Karena itu klaim penelitian ini **bukan** \"modelnya akurat\", "
            "melainkan \"modelnya andal **pada kelas yang batasnya tegas**\"."
        )
    with c2:
        st.subheader("Lapis 2 — Class Weighting (Saat pelatihan)")
        w = pd.DataFrame({
            "Kelas": ["Kelas 0 · Non-CumLaude", "Kelas 1 · CumLaude"],
            "Bobot": [round(136 / (2 * 54), 2), round(136 / (2 * 82), 2)],
        })
        fig = px.bar(w, x="Kelas", y="Bobot", text="Bobot", color="Kelas",
                     color_discrete_sequence=[RED, GREEN])
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="bobot netral (1,0)")
        fig.update_layout(showlegend=False, height=220, margin=dict(t=20, b=10),
                          xaxis_title="", yaxis_title="Bobot penalti")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Mekanisme.** Bobot dihitung *balanced*: `w = n / (k × n_kelas)`.\n\n"
            "- Kelas 0 → 136 / (2 × 54) = **1,26**\n"
            "- Kelas 1 → 136 / (2 × 82) = **0,83**\n\n"
            "**Efek pada fungsi loss.** Satu kesalahan pada mahasiswa berisiko dihukum **≈ 1,5× lebih berat** "
            "daripada kesalahan pada mahasiswa aman. Model kehilangan insentif untuk \"malas\" menebak mayoritas.\n\n"
            "**Kenapa cost-sensitive, bukan SMOTE?** Pada n = 136, oversampling sintetis berisiko mengarang "
            "mahasiswa yang tidak pernah ada dan memicu kebocoran antar-fold. Class weighting **tidak menambah "
            "satu baris data pun**, hanya mengubah harga kesalahan. Lebih aman untuk sampel kecil."
        )

    st.divider()
    st.subheader("Lapis 3 — Validasi & Metrik yang Tahan Imbalance (Pasca-pelatihan)")
    a, b, c = st.columns(3)
    a.info(
        "**Stratified RSKF 5×3**\n\nProporsi 39,7 : 60,3 dijaga **identik di seluruh 15 fold**. "
        "Tanpa stratifikasi, satu fold bisa kebetulan hanya berisi segelintir kelas minoritas dan "
        "menghasilkan metrik yang menipu."
    )
    b.info(
        "**Threshold tuning per fold**\n\nCut-off keputusan **dioptimalkan per fold (rentang 0,42–0,60)**, "
        "bukan default 0,50 yang secara diam-diam berpihak ke kelas mayoritas. "
        "Threshold tata kelola final LSTM = **0,46**."
    )
    c.info(
        "**Metrik robust**\n\nEvaluasi memakai **Balanced Accuracy, F1-Macro, dan MCC** — bukan akurasi mentah. "
        "Akurasi mentah bisa \"menang\" 60,3% hanya dengan menebak mayoritas; MCC tidak bisa dibohongi begitu."
    )

    st.subheader("Bukti Empiris: Accuracy vs Balanced Accuracy per Fold")
    st.caption(
        "Jika penanganan imbalance gagal, Accuracy akan melambung jauh di atas Balanced Accuracy — "
        "tanda model menang hanya karena menebak mayoritas. Jarak keduanya adalah alat ukurnya."
    )
    fnn = load("fnn_fold_results.csv")
    lstm = load("lstm_fold_results.csv")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**FNN**")
        st.line_chart(fnn.set_index("Fold")[["Accuracy", "Balanced_Accuracy"]])
    with d2:
        st.markdown("**BiLSTM + Attention (Sekuens Tematik)**")
        st.line_chart(lstm.set_index("Fold")[["Accuracy", "Balanced_Accuracy"]])

    gap_fnn = (fnn["Accuracy"].mean() - fnn["Balanced_Accuracy"].mean()) * 100
    gap_lstm = (lstm["Accuracy"].mean() - lstm["Balanced_Accuracy"].mean()) * 100
    st.success(
        f"**Diagnosis:** selisih rata-rata Accuracy − Balanced Accuracy hanya **{gap_fnn:+.1f} poin persentase (FNN)** "
        f"dan **{gap_lstm:+.1f} poin persentase (BiLSTM)** di seluruh 15 fold.\n\n"
        f"Selisih sekecil ini berarti model **memperoleh akurasinya dari kedua kelas secara merata**, bukan dengan "
        f"mengorbankan kelas minoritas. Untuk sebuah *early warning system*, ini justru properti terpenting: "
        f"sistem yang buta terhadap mahasiswa berisiko adalah sistem yang gagal total, sekalipun akurasi globalnya tinggi."
    )

# ================================================================ 3. KOMPARASI
elif page.startswith("3"):
    st.title("Evaluasi Komparasi Kinerja Prediktif — FNN vs BiLSTM + Attention")
    st.caption(
        "Seluruh angka dibaca langsung dari artefak Governance Cross-Validation (15 fold RSKF 5×3). "
        "Batang galat menunjukkan simpangan baku antar-fold."
    )

    fnn = load("fnn_fold_results.csv")
    lstm = load("lstm_fold_results.csv")
    metrics = ["Accuracy", "F1", "ROC_AUC", "Balanced_Accuracy", "MCC"]

    comp = pd.DataFrame({
        "Metrik": metrics,
        "FNN": [fnn[m].mean() for m in metrics],
        "LSTM (Sekuens Tematik)": [lstm[m].mean() for m in metrics],
    })
    fig = go.Figure()
    fig.add_bar(name="FNN", x=comp["Metrik"], y=comp["FNN"], marker_color=BLUE,
                text=comp["FNN"].round(3), textposition="outside",
                error_y=dict(type="data", array=[fnn[m].std() for m in metrics]))
    fig.add_bar(name="BiLSTM + Attention", x=comp["Metrik"],
                y=comp["LSTM (Sekuens Tematik)"], marker_color=AMBER,
                text=comp["LSTM (Sekuens Tematik)"].round(3), textposition="outside",
                error_y=dict(type="data", array=[lstm[m].std() for m in metrics]))
    fig.update_layout(barmode="group", height=440, yaxis_range=[0, 1.0],
                      yaxis_title="Skor (0–1)",
                      title="Rata-rata metrik lintas 15 fold (± simpangan baku)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analisis Temuan Struktural")
    st.markdown(
        f"1. **Fenomena Batas Langit-Langit Sinyal (*Data Signal Ceiling*):** Seluruh arsitektur dari baseline "
        f"hingga BiLSTM + Attention mentok pada pita **akurasi ± {min(comp['FNN'][0], comp['LSTM (Sekuens Tematik)'][0]):.1%}–{max(comp['FNN'][0], comp['LSTM (Sekuens Tematik)'][0]):.1%}** "
        f"dan **ROC-AUC ± 0,55–0,61**. Konvergensi seragam ini bukan kegagalan pemodelan, melainkan **karakteristik "
        f"data itu sendiri**: faktor eksternal sosiokultural inheren dipenuhi derau (*noise*), dan 40 fitur dari "
        f"136 sampel memang hanya membawa sinyal sebanyak itu. Menambah lapisan atau parameter tidak akan menembusnya.\n"
        f"2. **Simpangan baku antar-fold yang lebar:** Batang galat pada grafik mengungkap risiko yang sering "
        f"disembunyikan angka rata-rata: pada n = 136, **posisi juara bisa berpindah tergantung fold mana yang kebetulan "
        f"jadi data uji**. Karena itu klaim keunggulan di penelitian ini selalu dilaporkan bersama sebaran variansinya, "
        f"bukan sekadar nilai tengah.\n"
        f"3. **Implikasi pemilihan model:** Ketika dua arsitektur berhimpitan di dalam pita variansi yang sama, "
        f"kriteria pemilihan bergeser dari \"siapa paling akurat\" menjadi **\"siapa paling stabil dan paling "
        f"terkalibrasi\"** dan di situlah dasar keputusan pada perbandingan arsitektur di bawah."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lintasan performa per fold — FNN")
        st.line_chart(fnn.set_index("Fold")[["Accuracy", "F1", "ROC_AUC", "MCC"]])
        st.error(
            f"**Signal Ceiling.** Akurasi CV FNN stagnan di **{fnn['Accuracy'].mean():.2%}** dengan variansi lebar "
            f"(SD = ±{fnn['Accuracy'].std():.4f}). Lapisan *dense* menangkap interaksi non-linear antar-fitur, "
            f"tetapi tanpa struktur input ia memperlakukan 40 fitur sebagai 40 kolom yang setara — konteks antar-domain hilang."
        )
    with c2:
        st.subheader("Lintasan performa per fold — BiLSTM + Attention")
        st.line_chart(lstm.set_index("Fold")[["Accuracy", "F1", "ROC_AUC", "MCC"]])
        st.success(
            f"**Efek Sekuens Tematik.** Penataan 8 domain menstabilkan konvergensi antar-fold "
            f"(SD Accuracy = ±{lstm['Accuracy'].std():.4f}). Attention memberi bobot adaptif per domain, "
            f"sehingga model bisa \"memutuskan\" domain mana yang paling menentukan untuk tiap mahasiswa — "
            f"kemampuan yang tidak dimiliki FNN."
        )

    st.divider()
    st.subheader("Perbandingan Arsitektur Sekuensial (RSKF 5×5 · 25 fold)")
    st.caption(
        "Tiga arsitektur sekuensial diuji pada Sekuens Tematik 8 domain yang identik. "
        "Tujuannya: membuktikan bahwa keunggulan berasal dari **arsitekturnya**, bukan dari kebetulan pembagian data."
    )

    arch = pd.DataFrame({
        "Arsitektur": ["LSTM Standar", "BiLSTM + Attention", "Transformer Encoder"],
        "Akurasi": [0.6400, 0.6736, 0.6327],
        "F1-Macro": [0.6277, 0.6553, 0.6240],
        "F1 Kelas 0": [0.5896, 0.5917, 0.5935],
        "F1 Kelas 1": [0.6657, 0.7190, 0.6545],
    })
    arch_long = arch.melt("Arsitektur", var_name="Metrik", value_name="Skor")
    fig = px.bar(
        arch_long, x="Metrik", y="Skor", color="Arsitektur", barmode="group",
        color_discrete_map={"LSTM Standar": TEAL, "BiLSTM + Attention": AMBER,
                            "Transformer Encoder": PURPLE},
        text_auto=".3f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(t=30, b=10), yaxis_range=[0, 0.85],
                      yaxis_title="Skor (0–1)", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    b1, b2, b3 = st.columns(3)
    b1.info(
        "**LSTM Standar (baseline)**\n\nMembaca 8 domain hanya **satu arah** (t1 → t8). Domain awal tidak pernah "
        "melihat konteks domain akhir, padahal urutan domain di sini bersifat tematik dan bukan kronologis, "
        "sehingga arah baca tunggal justru membuang informasi.\n\n**Akurasi 64,00%.**"
    )
    b2.success(
        "**🏆 BiLSTM + Attention - arsitektur final**\n\nMembaca sekuens **dua arah** (maju & mundur), lalu "
        "*attention* memberi **bobot adaptif** pada domain paling informatif. Unggul di **seluruh metrik agregat**, "
        "dengan **F1 Kelas 1 tertinggi (0,719)**.\n\n**Akurasi 67,36%.**"
    )
    b3.warning(
        "**Transformer Encoder**\n\nSelf-attention penuh tidak membawa *inductive bias* sekuensial apa pun "
        "maka ia harus **mempelajari** struktur urutan dari data. Pada n = 136 itu jelas mustahil, dan hasilnya "
        "overfit. Skornya bahkan **di bawah baseline**.\n\n**Akurasi 63,27%.**"
    )

    st.markdown(
        "**Kenapa temuan Transformer justru memperkuat argumen penelitian:** kegagalannya membuktikan bahwa "
        "yang dibutuhkan data kecil ini **bukan kapasitas model yang lebih besar**, melainkan **struktur induktif "
        "yang tepat**. Sekuens Tematik + BiLSTM memberi model *prior* tentang bagaimana domain kehidupan saling "
        "terkait. Transformer harus menebaknya sendiri dari 136 sampel dan hasilnya gagal."
    )

    st.dataframe(pd.DataFrame({
        "Arsitektur": ["LSTM Standar (baseline)", "🏆 BiLSTM + Attention", "Transformer Encoder"],
        "Akurasi": ["64,00%", "67,36%", "63,27%"],
        "F1-Macro": [0.6277, 0.6553, 0.6240],
        "F1 Kelas 0": [0.5896, 0.5917, 0.5935],
        "F1 Kelas 1": [0.6657, 0.7190, 0.6545],
        "Keputusan": ["Pembanding", "Arsitektur final", "Pembanding"],
    }), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Catatan Transparansi: Hold-out 28 Sampel & Sensitivitas Threshold")
    hold = load("lstm_classification_report.csv")
    acc_default = float(hold.loc[hold["Class"] == "accuracy", "precision"].iloc[0])
    st.warning(
        f"Artefak `lstm_classification_report.csv` mencatat akurasi hold-out **{acc_default:.2%}** pada "
        f"**threshold default 0,50**, sedangkan angka **71,43%** yang dikutip di laporan berasal dari "
        f"**threshold tata kelola 0,46** hasil optimasi per fold.\n\n"
        f"**Keduanya benar — yang berbeda adalah cut-off keputusannya.** Pada hold-out sekecil 28 sampel, "
        f"pergeseran threshold sebesar 0,04 sudah cukup memindahkan beberapa sampel melintasi garis batas dan "
        f"mengubah akurasi secara mencolok. Justru inilah alasan penelitian ini **menjadikan metrik 15-fold CV "
        f"sebagai hasil utama**, bukan hold-out: pada n = 28, satu sampel bernilai 3,6 poin persentase, "
        f"sehingga angkanya terlalu rapuh untuk dijadikan klaim.",
        icon="⚠️",
    )

# ================================================================ 4. FITUR
elif page.startswith("4"):
    st.title("Audit Faktor Penggerak Eksternal Utama (Feature Governance)")
    st.caption(
        "Permutation Feature Importance — diagregasi lintas 15 fold, dibaca langsung dari artefak governance. "
        "Metode ini dipilih sebagai padanan neural network atas *native feature importance* milik Random Forest, "
        "agar perbandingan lintas-arsitektur tetap adil."
    )

    st.info(
        "**Cara kerja Permutation Importance.** Nilai satu fitur diacak secara acak, lalu diukur "
        "**seberapa jauh performa model jatuh**. Semakin besar penurunannya, semakin bergantung model pada fitur itu. "
        "Keunggulannya: metode ini **agnostik terhadap arsitektur**, Karena bisa diterapkan pada FNN maupun BiLSTM dengan "
        "prosedur yang persis sama, sehingga peringkat keduanya benar-benar sebanding."
    )

    model = st.radio("Pilih arsitektur yang diaudit:", ["LSTM (Sekuens Tematik)", "FNN"], horizontal=True)
    file_drv = ("lstm_external_driver_summary.csv" if model.startswith("LSTM")
                else "fnn_external_driver_summary.csv")
    drv_full = load(file_drv)
    drv = drv_full.head(12)

    fig = px.bar(
        drv.sort_values("Importance_Mean"),
        x="Importance_Mean", y="Feature", orientation="h",
        color="Governance_Status",
        color_discrete_map={"Dominant Driver": RED, "Latent Core": GREEN,
                            "Regular Feature": BLUE},
        labels={"Importance_Mean": "Mean Permutation Importance", "Feature": "",
                "Governance_Status": "Status Tata Kelola"},
        text_auto=".4f",
    )
    fig.update_layout(height=520, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Kamus Status Tata Kelola Fitur")
    g1, g2, g3 = st.columns(3)
    g1.error(
        "**🔴 Dominant Driver**\n\nKontribusi tertinggi terhadap keputusan model. Fitur inilah yang paling "
        "menentukan apakah seorang mahasiswa ditandai berisiko — sekaligus **titik ungkit intervensi paling efektif** "
        "bagi kampus."
    )
    g2.success(
        "**🟢 Latent Core**\n\nKontribusi rata-rata sedang, **tetapi dengan variabilitas tinggi antar-fold** "
        "(CV besar). Artinya fitur ini **sangat menentukan untuk sebagian mahasiswa dan nyaris tidak berarti "
        "bagi yang lain** — sinyal personal, bukan sinyal populasi. Justru fitur seperti inilah yang paling "
        "berbahaya bila diotomatisasi tanpa pengawasan manusia."
    )
    g3.info(
        "**🔵 Regular Feature**\n\nKontribusi konsisten namun moderat. Dipertahankan dalam model karena "
        "menyumbang stabilitas, tetapi **tidak layak dijadikan dasar tunggal** untuk memicu intervensi kampus."
    )

    st.divider()
    st.subheader("Divergensi Fokus Pemrosesan Antar-Arsitektur")

    fnn_drv = load("fnn_external_driver_summary.csv").head(5)
    lstm_drv = load("lstm_external_driver_summary.csv").head(5)
    st.table(pd.DataFrame({
        "Peringkat": [1, 2, 3, 4, 5],
        "FNN (Permutation Importance)": fnn_drv["Feature"].tolist(),
        "Skor FNN": fnn_drv["Importance_Mean"].round(4).tolist(),
        "BiLSTM + Attention (Permutation Importance)": lstm_drv["Feature"].tolist(),
        "Skor BiLSTM": lstm_drv["Importance_Mean"].round(4).tolist(),
    }))

    st.markdown(
        f"**Dua arsitektur, dua cara memandang mahasiswa yang sama — dan justru di situ letak temuannya.**\n\n"
        f"- **FNN** mengunci **`{fnn_drv['Feature'].iloc[0]}`** ({fnn_drv['Importance_Mean'].iloc[0]:.4f}) sebagai "
        f"penggerak dominan. Sebagai model tanpa struktur sekuensial, FNN mencari **sinyal tunggal terkuat** yang "
        f"berkorelasi non-linear dengan capaian akademik — dan status kemandirian ekonomi adalah pembeda paling tajam "
        f"yang bisa ia temukan dalam ruang fitur datar.\n"
        f"- **BiLSTM + Attention** justru menempatkan **`{lstm_drv['Feature'].iloc[0]}`** "
        f"({lstm_drv['Importance_Mean'].iloc[0]:.4f}) di puncak. Karena membaca domain secara berurutan, model ini "
        f"memperlakukan **kualitas lingkungan fisik belajar sebagai *gatekeeper* di hulu sekuens**: kondisi ruang "
        f"dan fasilitas yang buruk menyetel ulang cara seluruh domain berikutnya dibaca.\n"
        f"- **Konvergensi yang paling meyakinkan:** meski peringkat teratasnya berbeda, kedua arsitektur "
        f"**sama-sama menaikkan blok Angkatan (`Angkatan_2022`, `Angkatan_2024`)** ke jajaran atas. Konsensus "
        f"lintas-arsitektur ini menguatkan dugaan adanya **efek kohort/temporal** — angkatan yang berbeda menghadapi "
        f"kondisi eksternal yang berbeda, dan model mendeteksinya secara independen."
    )

    st.divider()
    st.subheader("Implikasi Kebijakan: Dari Peringkat Fitur ke Meja Intervensi")
    st.caption(
        "Feature importance hanya berguna jika bisa diterjemahkan menjadi tindakan. Tiga kanal berikut "
        "dipetakan langsung dari domain penggerak teratas."
    )
    a, b, c = st.columns(3)
    a.info(
        "**Kanal 1 — Beban kerja & manajemen waktu**\n\nDipicu oleh dominasi `Status_Bekerja` dan frekuensi absen "
        "akibat kegiatan luar kampus.\n\n**Tindakan:** konseling penjadwalan, dan yang lebih penting — "
        "**pertanyakan apakah mahasiswa bekerja karena pilihan atau karena terpaksa.** Bila terpaksa, ini sebenarnya "
        "masalah finansial yang menyamar sebagai masalah manajemen waktu."
    )
    b.info(
        "**Kanal 2 — Tekanan ekonomi & dukungan keluarga**\n\nDipicu oleh `Pengaruh_Ekonomi_Keluarga`, `Uang_Saku`, "
        "dan `Duk_Akademik_Keluarga`.\n\n**Tindakan:** penyaluran beasiswa/keringanan yang **proaktif** — sistem "
        "menghubungi mahasiswa, bukan menunggu mahasiswa mengajukan. Mahasiswa dengan tekanan tertinggi justru "
        "yang paling jarang mengajukan."
    )
    c.info(
        "**Kanal 3 — Fasilitas & lingkungan fisik**\n\nDipicu oleh `Jenis_Fa_Ruang_Kelas` — penggerak nomor satu "
        "versi BiLSTM.\n\n**Tindakan:** bantuan logistik (ruang belajar, perangkat, akses internet). Ini kanal "
        "**paling murah dan paling cepat dieksekusi** kampus, sekaligus yang paling sering luput dari radar "
        "sistem monitoring akademik konvensional."
    )

# ================================================================ 5. SIMULASI
elif page.startswith("5"):
    st.title("Simulasi Triase Prediksi — Early Warning System")
    st.caption(
        "Demonstrasi alur keputusan end-to-end: profil eksternal masuk → probabilitas & entropi keluar → "
        "koridor tata kelola menentukan apakah hasil boleh diotomatisasi atau wajib ditinjau manusia."
    )
    st.warning(
        "**Mode demo (mockup).** Skor dihitung dari pembobotan *permutation importance* model BiLSTM terhadap input, "
        "**bukan** inferensi langsung file `.keras`. Arah dan urutan bobotnya mengikuti temuan penelitian, sehingga "
        "**perilaku sistemnya** sahih untuk didemonstrasikan — yang tidak sahih adalah **angka presisinya**. "
        "Pada implementasi penuh, blok skor ini diganti pemanggilan pipeline preprocessing + model final.",
        icon="⚠️",
    )

    with st.form("form_ews"):
        st.subheader("1 · Postur Entri Faktor Eksternal")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Domain Finansial & Keluarga**")
            ekonomi = st.select_slider("Tekanan ekonomi keluarga",
                                       ["Sangat ringan", "Ringan", "Sedang", "Berat", "Sangat berat"], "Sedang")
            duk_keluarga = st.select_slider("Dukungan akademik keluarga",
                                            ["Sangat rendah", "Rendah", "Sedang", "Tinggi", "Sangat tinggi"], "Tinggi")
            uang_saku = st.select_slider("Kecukupan uang saku bulanan",
                                         ["Sangat kurang", "Kurang", "Cukup", "Lebih"], "Cukup")
        with c2:
            st.markdown("**Domain Manajemen Diri & Pekerjaan**")
            waktu = st.select_slider("Kualitas manajemen waktu",
                                     ["Sangat buruk", "Buruk", "Sedang", "Baik", "Sangat baik"], "Sedang")
            absen = st.slider("Frekuensi absen krn kegiatan eksternal (per bulan)", 0, 10, 2)
            kerja = st.selectbox("Status bekerja paruh waktu", ["Tidak bekerja", "≤ 20 jam/minggu", "> 20 jam/minggu"])
        with c3:
            st.markdown("**Domain Fasilitas & Psikososial**")
            fasilitas = st.select_slider("Kelengkapan fasilitas belajar",
                                         ["Sangat minim", "Minim", "Cukup", "Lengkap"], "Cukup")
            wellbeing = st.slider("Skor kesejahteraan psikososial (wellbeing)", 1, 10, 6)
            tidur = st.slider("Rata-rata jam tidur per hari", 3, 10, 6)
        submitted = st.form_submit_button("🔍 Jalankan Prediksi", use_container_width=True)

    if submitted:
        # --- skor demo: bobot mengikuti arah temuan permutation importance BiLSTM
        s = 0.0
        s += {"Sangat ringan": .10, "Ringan": .05, "Sedang": 0, "Berat": -.09, "Sangat berat": -.14}[ekonomi]
        s += {"Sangat rendah": -.10, "Rendah": -.05, "Sedang": 0, "Tinggi": .06, "Sangat tinggi": .10}[duk_keluarga]
        s += {"Sangat kurang": -.06, "Kurang": -.03, "Cukup": .02, "Lebih": .04}[uang_saku]
        s += {"Sangat buruk": -.16, "Buruk": -.09, "Sedang": 0, "Baik": .09, "Sangat baik": .15}[waktu]
        s += -.02 * max(absen - 2, 0)
        s += {"Tidak bekerja": .03, "≤ 20 jam/minggu": -.03, "> 20 jam/minggu": -.09}[kerja]
        s += {"Sangat minim": -.11, "Minim": -.06, "Cukup": .02, "Lengkap": .08}[fasilitas]
        s += .012 * (wellbeing - 5)
        s += .012 * (tidur - 6)
        prob = float(1 / (1 + np.exp(-4.0 * s)))          # prob kelas 1 (CumLaude)
        conf = max(prob, 1 - prob)
        entropy = float(-(prob * np.log2(prob + 1e-9) + (1 - prob) * np.log2(1 - prob + 1e-9)))
        thr = 0.46                                         # threshold governance BiLSTM

        st.divider()
        st.subheader("2 · Diagnosis Probabilitas & Ketidakpastian")
        k1, k2, k3 = st.columns([1.2, 1, 1])
        label = "CumLaude (Kelas 1)" if prob >= thr else "Non-CumLaude / Berisiko (Kelas 0)"
        k1.metric("Prediksi kategori IPK", label)
        k2.metric("Probabilitas CumLaude", f"{prob:.1%}", f"threshold tata kelola {thr:.2f}")
        k3.metric("Confidence · Entropy", f"{conf:.1%}", f"H = {entropy:.2f} bit")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": NAVY},
                   "steps": [{"range": [0, thr * 100], "color": "#F5D0C5"},
                             {"range": [thr * 100, 100], "color": "#CDE7CF"}],
                   "threshold": {"line": {"color": RED, "width": 3}, "value": thr * 100}},
            title={"text": "Probabilitas kelulusan CumLaude"},
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"**Membaca dua angka yang sama sekali berbeda maknanya.**\n\n"
            f"- **Probabilitas ({prob:.1%})** menjawab *\"model menebak apa?\"* — dibandingkan terhadap "
            f"threshold **{thr:.2f}**, bukan 0,50. Threshold ini bukan angka sembarangan: ia hasil optimasi "
            f"per fold, dan sengaja **digeser ke bawah 0,50 agar sistem lebih mudah menandai mahasiswa berisiko**. "
            f"Dalam konteks EWS, *false alarm* hanya berbiaya satu sesi konseling; *miss* berbiaya satu mahasiswa "
            f"yang terlambat ditolong.\n"
            f"- **Entropi ({entropy:.2f} bit)** menjawab pertanyaan yang jauh lebih penting: *\"seberapa yakin model "
            f"pada tebakannya sendiri?\"* Nilainya bergerak dari **0 bit** (model sepenuhnya yakin) hingga "
            f"**1 bit** (model benar-benar bingung, setara melempar koin). Entropi tinggi **tidak berarti "
            f"mahasiswa aman atau berisiko** — ia berarti **model tidak layak dipercaya untuk kasus ini**."
        )

        st.divider()
        st.subheader("3 · Koridor Keputusan Operasional (Lapisan Tata Kelola)")
        if conf >= 0.75 and entropy <= 0.75:
            st.success(
                f"🟢 **STATUS: SAFE TO AUTOMATE**\n\n"
                f"**Justifikasi.** Probabilitas tajam (confidence = {conf:.1%} ≥ 75%) dan entropi rendah "
                f"(H = {entropy:.2f} ≤ 0,75). Model berada jauh dari garis batas keputusan, dan sinyal inputnya "
                f"konsisten.\n\n"
                f"**Tindakan sistem.** Hasil boleh langsung masuk dashboard monitoring otomatis tanpa "
                f"intervensi manual administrasi."
            )
        elif conf >= 0.60:
            st.warning(
                f"🟡 **STATUS: BOUNDARY ZONE — TINJAU BERKALA**\n\n"
                f"**Justifikasi.** Confidence {conf:.1%} dan entropi H = {entropy:.2f} menempatkan sampel ini "
                f"**dekat dengan garis batas keputusan** (*decision boundary confusion*). Prediksinya condong ke "
                f"satu arah, tetapi tidak cukup tegas untuk dipercaya sendirian.\n\n"
                f"**Tindakan sistem.** Masuk daftar pantau. Tidak memicu intervensi otomatis, tetapi ditinjau "
                f"ulang pada siklus monitoring berikutnya.",
                icon="⚠️",
            )
        else:
            st.error(
                f"🔴 **STATUS: HUMAN-IN-THE-LOOP REQUIRED**\n\n"
                f"**Justifikasi.** Entropi menumpuk di zona kritis (H = {entropy:.2f}) dengan confidence hanya "
                f"{conf:.1%} — model praktis **melempar koin** untuk kasus ini.\n\n"
                f"**Tindakan sistem.** Sesuai kerangka kerja **NIST AI RMF**, luaran otomatis **DITAHAN** dan "
                f"kasus diserahkan ke penilaian manual Dosen Pembimbing Akademik. Sistem yang jujur mengakui "
                f"ketidaktahuannya jauh lebih berguna daripada sistem yang percaya diri dan salah."
            )

        if prob < thr:
            st.subheader("4 · Rekomendasi Intervensi Preventif")
            recs = []
            if waktu in ("Sangat buruk", "Buruk") or absen > 4:
                recs.append("Konseling **manajemen waktu** & penjadwalan ulang beban kegiatan eksternal.")
            if ekonomi in ("Berat", "Sangat berat") or kerja == "> 20 jam/minggu":
                recs.append("Asesmen **bantuan finansial / beasiswa** dan penyesuaian jam kerja paruh waktu.")
            if fasilitas in ("Sangat minim", "Minim"):
                recs.append("Bantuan **fasilitas belajar** (perangkat, akses internet, ruang belajar).")
            if wellbeing <= 4 or tidur <= 5:
                recs.append("Rujukan layanan **kesejahteraan psikososial** kampus.")
            if not recs:
                recs.append("Pemanggilan konseling umum oleh dosen wali untuk pendalaman profil.")
            for r in recs:
                st.markdown(f"- {r}")

        st.caption(
            "⚖️ **Batas etis sistem.** Prediksi ini adalah dasar **prioritas pendampingan**, bukan vonis akademik, "
            "bukan dasar seleksi, dan bukan label yang boleh melekat pada mahasiswa. Sistem menentukan **siapa yang "
            "dihubungi lebih dahulu** — tidak pernah menentukan siapa yang layak."
        )

# ================================================================ 6. RISIKO
else:
    st.title("Tata Kelola Risiko AI — Risk Governance Funnel")
    st.caption(
        "Corong penyaringan seluruh prediksi lintas 15 fold CV, dibaca dari artefak governance asli. "
        "Kerangka acuan: NIST AI Risk Management Framework."
    )

    st.info(
        "**Kenapa corong ini ada.** Model dengan akurasi 67% berarti **satu dari tiga prediksinya salah** — dan "
        "model itu sendiri tidak tahu yang mana. Corong ini adalah mekanisme yang menjawabnya: alih-alih memaksakan "
        "keputusan biner pada semua sampel, sistem **menyaring keluar prediksi yang tidak layak dipercaya** dan "
        "menyerahkannya ke manusia. Yang diaudit bukan lagi akurasi model, melainkan **kejujuran model tentang "
        "ketidaktahuannya sendiri**."
    )

    st.subheader("Definisi Zona Penyaringan")
    z1, z2, z3, z4 = st.columns(4)
    z1.success("**Total Predictions**\n\nSeluruh prediksi yang dihasilkan lintas 15 fold — populasi awal corong.")
    z2.info("**Confidence Risk**\n\nProbabilitas tidak cukup tajam; model condong ke satu kelas tetapi tanpa keyakinan memadai.")
    z3.warning("**Entropy Risk**\n\nEntropi biner tinggi — sebaran probabilitas nyaris merata antar-kelas.")
    z4.error("**Critical Uncertainty**\n\n**Gagal di kedua kriteria.** Wajib diisolasi ke koridor Human-in-the-Loop.")

    st.divider()
    c1, c2 = st.columns(2)
    funnels = {}
    for col, name, file in [(c1, "FNN", "fnn_governance_funnel.csv"),
                            (c2, "BiLSTM + Attention (Sekuens Tematik)", "lstm_governance_funnel.csv")]:
        fun = load(file)
        funnels[name] = fun
        fig = go.Figure(go.Funnel(
            y=fun["Uncertainty_Zone"], x=fun["Sample_Count"],
            marker={"color": [GREEN, AMBER, "#D35400", RED]},
            textinfo="value+percent initial"))
        fig.update_layout(title=name, height=380, margin=dict(t=50, b=10))
        col.plotly_chart(fig, use_container_width=True)

    f_fnn = funnels["FNN"].set_index("Uncertainty_Zone")["Sample_Count"]
    f_lstm = funnels["BiLSTM + Attention (Sekuens Tematik)"].set_index("Uncertainty_Zone")["Sample_Count"]
    crit_fnn = f_fnn["Critical Uncertainty"] / f_fnn["Total Predictions"]
    crit_lstm = f_lstm["Critical Uncertainty"] / f_lstm["Total Predictions"]

    st.subheader("Analisis Temuan Tata Kelola")
    st.markdown(
        f"1. **Beban Human-in-the-Loop terukur, bukan diperkirakan.** Dari **{f_fnn['Total Predictions']} prediksi**, "
        f"FNN menyisakan **{f_fnn['Critical Uncertainty']} sampel ({crit_fnn:.1%})** di zona *Critical Uncertainty*, "
        f"sedangkan BiLSTM **{f_lstm['Critical Uncertainty']} sampel ({crit_lstm:.1%})**. Angka ini bisa langsung "
        f"diterjemahkan ke **beban kerja nyata Dosen Pembimbing Akademik** — inilah bentuk konkret dari prinsip "
        f"*measurable risk* NIST AI RMF.\n"
        f"2. **BiLSTM lebih akurat, tetapi juga lebih banyak mengaku ragu.** Ia menyisakan zona kritis yang lebih besar "
        f"({crit_lstm:.1%} vs {crit_fnn:.1%}). Ini **bukan kelemahan** — model yang lebih peka terhadap struktur data "
        f"juga lebih peka terhadap kasus yang memang ambigu. **Model yang berbahaya justru yang selalu percaya diri.**\n"
        f"3. **Implikasi kebijakan.** Seluruh luaran zona *Critical Uncertainty* **wajib diisolasi** dari otomatisasi "
        f"dan masuk koridor manual. Kampus mendapat dua hal sekaligus: **daftar prioritas** (siapa dihubungi lebih dulu) "
        f"dan **daftar kehati-hatian** (siapa yang tidak boleh diputuskan mesin)."
    )

    st.divider()
    st.subheader("Kurva Kalibrasi Probabilitas")
    st.caption(
        "Kalibrasi menguji hal yang tidak diukur akurasi: **apakah angka probabilitas model bisa dipercaya "
        "sebagai probabilitas?** Jika model bilang \"70% CumLaude\" pada 100 mahasiswa, seharusnya ±70 di antaranya "
        "memang CumLaude. Garis putus-putus adalah kalibrasi sempurna."
    )
    d1, d2 = st.columns(2)
    for col, name, file in [(d1, "FNN", "fnn_calibration_curve.csv"),
                            (d2, "BiLSTM + Attention", "lstm_calibration_curve.csv")]:
        cal = load(file)
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Kalibrasi ideal",
                        line=dict(dash="dash", color="gray"))
        fig.add_scatter(x=cal["Bin_Confidence"], y=cal["Bin_Accuracy"],
                        mode="lines+markers", name=name, line=dict(color=BLUE))
        fig.update_layout(title=f"Calibration Curve — {name}", height=360,
                          xaxis_title="Rata-rata probabilitas prediksi",
                          yaxis_title="Frekuensi observasi aktual", margin=dict(t=50, b=10))
        col.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Cara membacanya:**\n"
        "- Kurva **di atas garis ideal** → model **terlalu pesimis** (*underconfident*): kenyataannya lebih banyak "
        "mahasiswa CumLaude daripada yang ia duga.\n"
        "- Kurva **di bawah garis ideal** → model **terlalu percaya diri** (*overconfident*). **Ini kondisi paling "
        "berbahaya** untuk EWS: model meyakinkan kampus bahwa seorang mahasiswa aman, padahal tidak.\n"
        "- **Reliability Error** pada artefak mengukur jarak vertikal rata-rata terhadap garis ideal. "
        "Kalibrasi **Platt/sigmoid** diterapkan pada fold yang menunjukkan penyimpangan besar — bukan untuk "
        "memperbaiki akurasi (kalibrasi tidak mengubah peringkat prediksi sama sekali), melainkan untuk membuat "
        "**angka probabilitasnya layak dijadikan dasar keputusan tata kelola** di halaman ini."
    )

    st.success(
        "**Penutup argumen sidang.** Kontribusi penelitian ini bukan terletak pada angka akurasi — *signal ceiling* "
        "sudah menutup jalan itu sejak awal, dan itu terbukti sebagai batas data, bukan batas usaha. Kontribusinya "
        "adalah **membuktikan bahwa faktor eksternal saja membawa sinyal yang nyata namun terbatas**, lalu "
        "**membangun lapisan tata kelola yang membuat sinyal terbatas itu tetap aman dipakai**: AI sebagai "
        "**instrumen triase preventif**, bukan pengambil keputusan final."
    )
