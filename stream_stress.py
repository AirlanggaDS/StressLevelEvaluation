import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

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


# data = model_data["data"]
# df = model_data["df"]

df = model_data["data"]
# df = model_data["df"]
df = pd.DataFrame(df.loc[:, ['Usia', 'Jenis Kelamin', 'Status', 'Bekerja', 'Jam Tidur Perhari', 'Kesibukan Perhari', 'Frequency of Anger', 'Lack of Control', 'Anxiety and Stress', 'Self-Efficacy', 'Perceived Control', 'Perceived Overwhelm', 'Feeling of Mastery', 'Cumulative Difficulty']])
# Calculate the total score and add it as a new column 'Score'
df['Score'] = df[['Frequency of Anger', 'Lack of Control', 'Anxiety and Stress', 'Self-Efficacy', 'Perceived Control', 'Perceived Overwhelm', 'Feeling of Mastery', 'Cumulative Difficulty']].sum(axis=1)

st.title("Stress Level Evaluation Dashboard")

# Display the data
st.subheader("Data Overview")
st.write(df.head())

# Data preprocessing and clustering
df['Jenis Kelamin'].replace({'Laki-laki':0, 'Perempuan':1}, inplace=True)
df['Status'].replace({'Pelajar':0,'Mahasiswa':1,'Lainnya':2}, inplace=True)
df['Bekerja'].replace({'Tidak':0,'Ya':1}, inplace=True)

# Normalize numerical data
numerical_cols = ['Usia','Jam Tidur Perhari', 'Kesibukan Perhari','Score']
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Feature selection
selected_features = ['Usia', 'Jenis Kelamin', 'Status', 'Bekerja', 'Jam Tidur Perhari','Kesibukan Perhari',
                     'Frequency of Anger', 'Lack of Control', 'Anxiety and Stress', 'Self-Efficacy',
                     'Perceived Control', 'Perceived Overwhelm', 'Feeling of Mastery', 'Cumulative Difficulty','Score']

X1 = df[selected_features]

# Clustering using K-Means
inertia = []
silhouette_scores = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X1)
    
    inertia.append(kmeans.inertia_)
    
    if len(set(kmeans.labels_)) > 1:  # Silhouette score requires at least 2 different clusters
        silhouette_scores.append(silhouette_score(X1, kmeans.labels_))
    else:
        silhouette_scores.append(None)

# Plot Elbow Method
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(2, 11), inertia, marker='o')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
st.pyplot(fig)


# Choose optimal number of clusters (example: 3)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=55)
df['Cluster'] = kmeans.fit_predict(X1)

# Cluster distribution
cluster_distribution = df.groupby('Cluster')[selected_features].apply(lambda x: x.describe())
st.write("\nDistribusi untuk Setiap Cluster:")
st.write(cluster_distribution)

# Label clusters
def label_clusters(row):
    if row['Cluster'] == 2:
        return 'Low Stress'
    elif row['Cluster'] == 0:
        return 'Stress Moderated'
    elif row['Cluster'] == 1:
        return 'High Perceived Stress'

df['Tingkat Stress'] = df.apply(label_clusters, axis=1)

st.write("\nHasil Clustering dengan Label Tingkatan Cluster:")
st.write(df)

# Display cluster member counts
cluster_counts = df['Cluster'].value_counts()
for cluster_num, count in cluster_counts.items():
    st.write(f"{label_clusters({'Cluster': cluster_num})}: {count} anggota")

# Visualization: Stress Score Distribution
st.subheader("Additional Visualizations")

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Score", hue="Tingkat Stress", bins=20, kde=True)
plt.title("Stress Score Distribution by Stress Category")
plt.xlabel("Score")
plt.ylabel("Frequency")
st.pyplot(plt)

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Usia", y="Score", hue="Tingkat Stress", data=df, s=100, alpha=0.8)
plt.title("Scatter Plot by Clustering")
plt.xlabel("Usia")
plt.ylabel("Score")
plt.legend(title="Cluster")
st.pyplot(plt)

# 3D Scatter plot
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")

# Manually assign colors to each cluster
colors = {'Low Stress': 'purple', 'High Perceived Stress': 'orange', 'Stress Moderated': 'green'}

scatter = ax.scatter(
    df["Usia"],
    df["Score"],
    df["Kesibukan Perhari"],
    c=df["Tingkat Stress"].map(colors),  # Use the manually assigned colors
    s=50,
    alpha=0.8,
)
ax.set_xlabel("Usia")
ax.set_ylabel("Score")
ax.set_zlabel("Kesibukan Perhari")
ax.set_title("3D Scatter Plot by Clustering")

# Create a custom legend outside the plot
legend_labels = list(colors.keys())
legend_markers = [plt.Line2D([0], [0], marker='o', color=colors[label], linestyle='', markersize=10) for label in legend_labels]
ax.legend(legend_markers, legend_labels, loc='upper left', title='Cluster')

st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")

# Exclude non-numeric columns from the correlation matrix
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Correlation Plot")
st.pyplot(plt)

# Data preprocessing
y = df["Tingkat Stress"]
X = df.drop(["Usia", "Jenis Kelamin", "Status", "Bekerja", "Jam Tidur Perhari", "Kesibukan Perhari", "Cluster", "Score", "Tingkat Stress"], axis=1)

# Replace numeric labels with categorical labels for clarity
df['Tingkat Stress'].replace({0: 'Low Stress', 1: 'Stress Moderated', 2: 'High Perceived Stress'}, inplace=True)

# Split the data into training and testing sets
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)

# Standardize the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model
model_rf = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=100, random_state=100)
model_rf.fit(X_train, y_train)

# Evaluate the model
predictions_test_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, predictions_test_rf)
precision_rf = precision_score(y_test, predictions_test_rf, average='weighted', zero_division=1)
recall_rf = recall_score(y_test, predictions_test_rf, average='weighted')
f1_rf = f1_score(y_test, predictions_test_rf, average='weighted')

# Display metrics in Streamlit
st.subheader("Random Forest Metrics")
st.write(f"Accuracy (Random Forest): {accuracy_rf}")
st.write(f"Precision (Random Forest): {precision_rf}")
st.write(f"Recall (Random Forest): {recall_rf}")
st.write(f"F1 Score (Random Forest): {f1_rf}")

# Plot the confusion matrix
st.subheader("Confusion Matrix Heatmap")
cm_rf = confusion_matrix(y_test, predictions_test_rf)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=model_rf.classes_, yticklabels=model_rf.classes_, ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix Heatmap')
st.pyplot(fig)

# Plot feature importances
st.subheader("Feature Importances in Random Forest")
feature_importances = pd.DataFrame(model_rf.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))  # Change 'fix' to 'fig'
sns.barplot(x=feature_importances.importance, y=feature_importances.index, ax=ax)  # Added ax=ax
ax.set_title('Feature Importances in Random Forest')
ax.set_xlabel('Relative Importance')
ax.set_ylabel('Feature')
st.pyplot(fig)

# Visualize a decision tree from the Random Forest (showing one tree, not the entire ensemble)
st.subheader("Random Forest Visualization")
fig, ax = plt.subplots(figsize=(20, 10))  # Change 'fig' to 'fig, ax'
plot_tree(model_rf.estimators_[0], feature_names=X.columns, class_names=['Low Stress', 'Stress Moderated', 'High Perceived Stress'], filled=True, rounded=True, ax=ax)  # Added ax=ax
ax.set_title("Random Forest Visualization", fontsize=15)
st.pyplot(fig)
