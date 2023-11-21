import pickle
import streamlit as st

# Membaca model
stress_model = pickle.load(open('stressLevel_logreg.sav','rb'))

# Judul Web
st.title('Stress Level Evaluation')

# Input
Usia = st.text_input('Masukkan usia Anda')
JenisKelamin = st.text_input('Jenis Kelamin')
Status = st.text_input('Status (Pelajar/Mahasiswa/Lainnya)')
### Bekerja', 'Jam Tidur Perhari', 'Kesibukan Perhari', 'Frequency of Anger', 'Lack of Control', 'Anxiety and Stress', 'Self-Efficacy', 'Perceived Control', 'Perceived Overwhelm', 'Feeling of Mastery', 'Cumulative Difficulty'

# Proses
## Prediksi
stress_cluster = ''

# Tombol prediksi
if st.button('Ukur Tingkat Stress'):
    stress_eval = stress_model.predict([[Usia, JenisKelamin, Status]])