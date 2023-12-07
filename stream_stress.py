import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the model and scaler
model_data = pickle.load(open("stressLevel.sav", "rb"))
model_rf = model_data["model"]
scaler = model_data["scaler"]

# Web Title
st.title("Stress Level Evaluation")

# Fungsi untuk konversi pilihan selectbox menjadi nilai numerik
def convert_selectbox_choice(choice):
    choices_map = {
        "Tidak pernah": 1,
        "Hampir tidak pernah": 2,
        "Terkadang": 3,
        "Cukup sering": 4,
        "Sangat sering": 5,
    }
    return choices_map.get(choice, 0)


option = [
    "Tidak pernah",
    "Hampir tidak pernah",
    "Terkadang",
    "Cukup sering",
    "Sangat sering",
]

FrequencyOfAnger = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda kesal karena suatu hal terjadi secara tak terduga?",
    option,
    key="anger",
)
FrequencyOfAnger = convert_selectbox_choice(FrequencyOfAnger)

LackOfControl = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda merasa tidak mampu mengendalikan hal-hal penting dalam hidup Anda?",
    option,
    key="control",
)
LackOfControl = convert_selectbox_choice(LackOfControl)

AnxietyAndStress = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda merasa gugup dan stres?",
    option,
    key="anxiety_stress",
)
AnxietyAndStress = convert_selectbox_choice(AnxietyAndStress)

SelfEfficacy = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda merasa yakin dengan kemampuan Anda dalam menangani masalah pribadi?",
    option,
    key="self_efficacy",
)
SelfEfficacy = convert_selectbox_choice(SelfEfficacy)

PerceivedControl = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda merasa segalanya berjalan sesuai keinginan Anda?",
    option,
    key="perceived_control",
)
PerceivedControl = convert_selectbox_choice(PerceivedControl)

PerceivedOverwhelm = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda menyadari bahwa Anda tidak mampu menyelesaikan semua hal yang harus Anda lakukan?",
    option,
    key="perceived_overwhelm",
)
PerceivedOverwhelm = convert_selectbox_choice(PerceivedOverwhelm)

FeelingOfMastery = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda merasa berada di puncak segalanya?",
    option,
    key="feeling_of_mastery",
)
FeelingOfMastery = convert_selectbox_choice(FeelingOfMastery)

CumulativeDifficulty = st.selectbox(
    "Dalam sebulan terakhir, seberapa sering Anda merasakan kesulitan yang menumpuk sehingga Anda tidak mampu mengatasinya?",
    option,
    key="cumulative_difficulty",
)
CumulativeDifficulty = convert_selectbox_choice(CumulativeDifficulty)

# Process
## Prediction
stress_predict = ""

if st.button("Ukur Tingkat Stress"):
    # Transform input data
    input_data = np.array(
        [
            FrequencyOfAnger,
            LackOfControl,
            AnxietyAndStress,
            SelfEfficacy,
            PerceivedControl,
            PerceivedOverwhelm,
            FeelingOfMastery,
            CumulativeDifficulty,
        ]
    ).reshape(1, -1)

    # Convert string input to numeric
    input_data = input_data.astype(float)  # Change data type to float

    st.write("Input Data:")
    st.write(
        pd.DataFrame(
            input_data,
            columns=[
                "Frequency of Anger",
                "Lack of Control",
                "Anxiety and Stress",
                "Self-Efficacy",
                "Perceived Control",
                "Perceived Overwhelm",
                "Feeling of Mastery",
                "Cumulative Difficulty",
            ],
        )
    )

    # Transform the input data using the saved scaler if available
    if scaler is not None:
        scaled_data = scaler.transform(input_data)
        st.write("Scaled Data:")
        st.write(
            pd.DataFrame(
                scaled_data,
                columns=[
                    "Frequency of Anger",
                    "Lack of Control",
                    "Anxiety and Stress",
                    "Self-Efficacy",
                    "Perceived Control",
                    "Perceived Overwhelm",
                    "Feeling of Mastery",
                    "Cumulative Difficulty",
                ],
            )
        )
    else:
        scaled_data = input_data  # If no scaler is available, use the original data

    # Predict using the scaled data
    prediction = model_rf.predict(scaled_data)
    st.write("Prediction:")
    st.write(pd.DataFrame(prediction, columns=["Stress Level"]))

    # Determine stress level based on prediction
    if prediction[0] == "High Perceived Stress":
        stress_cluster = "Stress Level Anda adalah Tinggi 😓"
        motivational_message = "Penting untuk diakui bahwa tingkat stres yang tinggi dapat menjadi beban yang berat. Anda tidak sendirian dalam perasaan ini, dan ada sumber dukungan yang dapat diakses. Pertimbangkan untuk berbagi dengan teman, keluarga, atau mencari bantuan profesional. Saya di sini untuk mendengarkan dan memberikan dukungan sepanjang perjalanan ini. 🤝"
    elif prediction[0] == "Stress Moderated":
        stress_cluster = "Stress Level Anda adalah Menengah 😌"
        motivational_message = "Saya mengakui bahwa hidup seringkali penuh dengan tantangan yang menuntut. Perasaan stres yang Anda alami adalah suatu hal yang wajar, dan kesadaran akan tingkat stres tersebut merupakan langkah penting. Pertimbangkan untuk menetapkan tujuan kecil dalam mengelola stres dan fokus pada langkah-langkah konkrit untuk mencapainya. Jika diperlukan, saya dapat memberikan panduan lebih lanjut. 🎯"
    else:
        stress_cluster = "Stress Level Anda adalah Rendah 😊"
        motivational_message = "Sangat membanggakan melihat bahwa tingkat stres Anda saat ini berada pada tingkat rendah. Pemeliharaan pola hidup sehat dan kelanjutan dari aktivitas yang memberikan kebahagiaan merupakan langkah positif yang luar biasa. Jangan ragu untuk berbagi pengalaman atau pertimbangan, saya siap mendengar dan memberikan dukungan. 🌈"

    # Display the stress level and motivational message in the Streamlit app
    st.success(stress_cluster)
    st.write(motivational_message)

    st.markdown(
        "*Jangan ragu untuk mengirimkan pesan jika ada yang ingin dibicarakan atau ditanyakan. Kami di sini untuk membantu, termasuk jika Anda ingin berbagi pengalaman atau mencari dukungan untuk meredakan stres. Silakan kirimkan pertanyaan atau komentar Anda, dan kami akan berusaha memberikan respons secepat mungkin.*",
        unsafe_allow_html=True,
    )
    # Menambahkan kontak di bawahnya
    st.write("📞 **Airlangga**: +62 889-9645-9159")
    st.write("📞 **Nathanael**: +62 812-8777-4009")
    st.write("📞 **Shabrina**: +62 857-0630-6913")
