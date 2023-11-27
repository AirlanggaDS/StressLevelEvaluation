import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the model and scaler
model_data = pickle.load(open('stressLevel.sav', 'rb'))
model_rf = model_data['model']
scaler = model_data['scaler']

# Web Title
st.title('Stress Level Evaluation')

# Input
# Usia = st.number_input('Masukkan usia Anda', min_value=0)
# JenisKelamin = st.radio('Jenis Kelamin', ['Laki-laki', 'Perempuan'], key='jenis_kelamin')
# Status = st.radio('Status', ['Pelajar', 'Mahasiswa', 'Lainnya'], key='status')
# Bekerja = st.radio('Apakah Anda bekerja (termasuk kerja paruh waktu)?', ['Ya', 'Tidak'], key='bekerja')
# JamTidurPerhari = st.number_input('Berapa jam Anda tidur dalam sehari? (contoh: 8)', min_value=0)
# KesibukanPerhari = st.number_input('Berapa jam kesibukan Anda dalam sehari? (bulatkan dalam jam)', min_value=0)

# Fungsi untuk konversi pilihan radio menjadi nilai numerik
def convert_radio_choice(choice):
    choices_map = {
        'Tidak pernah': 1,
        'Hampir tidak pernah': 2,
        'Terkadang': 3,
        'Cukup sering': 4,
        'Sangat sering': 5
    }
    return choices_map.get(choice, 0)

FrequencyOfAnger = st.radio('Dalam sebulan terakhir, seberapa sering Anda kesal karena suatu hal terjadi secara tak terduga?',
                            ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='anger')
FrequencyOfAnger = convert_radio_choice(FrequencyOfAnger)

LackOfControl = st.radio('Dalam sebulan terakhir, seberapa sering Anda merasa tidak mampu mengendalikan hal-hal penting dalam hidup Anda?',
                         ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='control')
LackOfControl = convert_radio_choice(LackOfControl)

AnxietyAndStress = st.radio('Dalam sebulan terakhir, seberapa sering Anda merasa gugup dan stres?',
                            ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='anxiety_stress')
AnxietyAndStress = convert_radio_choice(AnxietyAndStress)

SelfEfficacy = st.radio('Dalam sebulan terakhir, seberapa sering Anda merasa yakin dengan kemampuan Anda dalam menangani masalah pribadi?',
                         ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='self_efficacy')
SelfEfficacy = convert_radio_choice(SelfEfficacy)

PerceivedControl = st.radio('Dalam sebulan terakhir, seberapa sering Anda menyadari bahwa Anda tidak mampu menyelesaikan semua hal yang harus Anda lakukan?',
                            ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='perceived_control')
PerceivedControl = convert_radio_choice(PerceivedControl)

PerceivedOverwhelm = st.radio('Dalam sebulan terakhir, seberapa sering Anda menyadari bahwa Anda tidak mampu menyelesaikan semua hal yang harus Anda lakukan?',
                             ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='perceived_overwhelm')
PerceivedOverwhelm = convert_radio_choice(PerceivedOverwhelm)

FeelingOfMastery = st.radio('Dalam sebulan terakhir, seberapa sering Anda merasa berada di puncak segalanya?',
                            ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='feeling_of_mastery')
FeelingOfMastery = convert_radio_choice(FeelingOfMastery)

CumulativeDifficulty = st.radio('Dalam sebulan terakhir, seberapa sering Anda merasakan kesulitan yang menumpuk sehingga Anda tidak mampu mengatasinya?',
                                ['Tidak pernah', 'Hampir tidak pernah', 'Terkadang', 'Cukup sering', 'Sangat sering'], key='cumulative_difficulty')
CumulativeDifficulty = convert_radio_choice(CumulativeDifficulty)

# Process
## Prediction
stress_predict = '' 

if st.button('Ukur Tingkat Stress'):
    # Transform input data
    input_data = np.array([FrequencyOfAnger, LackOfControl, AnxietyAndStress, SelfEfficacy,
                           PerceivedControl, PerceivedOverwhelm, FeelingOfMastery, CumulativeDifficulty]).reshape(1, -1)

    # Convert string input to numeric
    input_data = input_data.astype(float)  # Change data type to float

    st.write("Input Data:")
    st.write(pd.DataFrame(input_data, columns=['Frequency of Anger', 'Lack of Control', 'Anxiety and Stress',
                                                'Self-Efficacy', 'Perceived Control', 'Perceived Overwhelm',
                                                'Feeling of Mastery', 'Cumulative Difficulty']))

    # Transform the input data using the saved scaler if available
    if scaler is not None:
        scaled_data = scaler.transform(input_data)
        st.write("Scaled Data:")
        st.write(pd.DataFrame(scaled_data, columns=['Frequency of Anger', 'Lack of Control', 'Anxiety and Stress',
                                                     'Self-Efficacy', 'Perceived Control', 'Perceived Overwhelm',
                                                     'Feeling of Mastery', 'Cumulative Difficulty']))
    else:
        scaled_data = input_data  # If no scaler is available, use the original data

    # Predict using the scaled data
    prediction = model_rf.predict(scaled_data)
    st.write("Prediction:")
    st.write(pd.DataFrame(prediction, columns=['Stress Level']))

    # Determine stress level based on prediction
    if prediction[0] == 'High Perceived Stress':
        stress_cluster = 'Stress Level Anda adalah High'
        motivational_message = 'Anda mungkin menghadapi banyak tekanan saat ini, tetapi ingatlah bahwa Anda memiliki kekuatan untuk mengatasinya. Ambil napas dalam-dalam dan fokus pada solusi.'
    elif prediction[0] == 'Stress Moderated':
        stress_cluster = 'Stress Level Anda adalah Moderated'
        motivational_message = 'Anda mungkin menghadapi beberapa tantangan, tetapi Anda dapat mengatasi mereka dengan kepala dingin. Percayalah pada diri sendiri dan ambil langkah-langkah kecil untuk mengurangi stres.'
    else:
        stress_cluster = 'Stress Level Anda adalah Low'
        motivational_message = 'Selamat! Tingkat stres Anda rendah. Tetap jaga keseimbangan hidup Anda dan terus lakukan hal-hal yang membuat Anda bahagia.'

    # Display the stress level and motivational message in the Streamlit app
    st.success(stress_cluster)
    st.write("Motivational Message:")
    st.write(motivational_message)

