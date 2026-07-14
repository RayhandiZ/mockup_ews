# Purwarupa Early Warning System — Prediksi Kategori IPK (FNN & LSTM)

Mockup demo sidang MBKM Penelitian — Rayhandi Zulmi (00000103940), Sistem Informasi UMN.

## Cara menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka http://localhost:8501 di browser.

## Struktur halaman (urutan storytelling sidang)
1. **Ringkasan Penelitian** — masalah, solusi, distribusi kelas pasca-pruning.
2. **Penanganan Imbalance** — pruning zona ambigu, class weighting, stratified RSKF + threshold tuning, bukti Accuracy vs Balanced Accuracy.
3. **Komparasi FNN vs LSTM** — metrik CV 15 fold dari artefak asli (Signal Ceiling vs Sekuens Tematik) + perbandingan arsitektur sekuensial (LSTM Standar vs BiLSTM+Attention vs Transformer, RSKF 5×5).
4. **Faktor Penggerak Utama** — permutation importance & implikasi intervensi.
5. **Simulasi Prediksi (Demo)** — form profil mahasiswa → probabilitas, gauge, triage risiko, rekomendasi.
6. **Tata Kelola Risiko AI** — Risk Governance Funnel & kurva kalibrasi.

## Catatan penting untuk sidang
- Grafik pada halaman 2, 3, 4, dan 6 membaca **CSV artefak governance asli** dari folder `data/`.
- Halaman 5 berjalan dalam **mode demo**: skor dihitung dari pembobotan permutation
  importance LSTM (arah temuan penelitian), **bukan** inferensi langsung file `.keras`.
  Untuk implementasi penuh, ganti blok skor pada `app.py` dengan pemanggilan
  pipeline preprocessing + model final.
- Threshold keputusan 0,46 mengikuti hasil Threshold Governance LSTM.
