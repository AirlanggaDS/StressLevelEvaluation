import pickle
import streamlit as st

# Membaca model
stress_model = pickle.load(open('stressLevel_logreg.sav','rb'))

# Judul Web
st.title('Stress Level Evaluation')

# Input
Usia = st.text_input('Masukkan usia Anda')
JenisKelamin = st.radio('Jenis Kelamin',
                        ['Laki-laki', 'Perempuan'])
Status = st.radio('Status',
                  ['Pelajar', 'Mahasiswa', 'Lainnya'])
Bekerja = st.radio('Apakah Anda bekerja (termasuk kerja paruh waktu)?', 
                        ['Ya', 'Tidak'])
JamTidurPerhari = st.text_input('Berapa jam Anda tidur dalam sehari? (contoh: 8)')
KesibukanPerhari = st.text_input('Berapa jam kesibukan Anda dalam sehari? (bulatkan dalam jam)')
FrequencyOfAnger = st.radio('Dalam sebulan terakhir, seberapa sering Anda kesal karena suatu hal terjadi secara tak terduga?',
                            ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering']
                            )
FrequencyOfAnger = st.radio('Dalam sebulan terakhir, seberapa sering Anda merasa tidak mampu mengendalikan hal-hal penting dalam hidup Anda?',
                            ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering']
                            )
### Bekerja', 'Jam Tidur Perhari', 'Kesibukan Perhari', 'Frequency of Anger', 'Lack of Control', 'Anxiety and Stress', 'Self-Efficacy', 'Perceived Control', 'Perceived Overwhelm', 'Feeling of Mastery', 'Cumulative Difficulty'

# Proses
## Prediksi
stress_predict = ''

# Tombol prediksi
if st.button('Ukur Tingkat Stress'):
    stress_predict = stress_model.predict([[Usia, JenisKelamin, Status, Bekerja, JamTidurPerhari, ]])

    if(stress_predict[0]==1): #ini apa coy
        stress_cluster = 'Stress Level Anda adalah Moderated'
    else :
        stress_cluster = 'Stress Level Anda adalah High'

    st.success(stress_cluster)
