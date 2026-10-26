import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
import io

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Demo Skripsi - Analisis Klaster AI Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling agar tampilan modern & profesional
st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-header { font-size: 15px; color: #4B5563; margin-bottom: 20px; }
    .step-box { background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #2563EB; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Sistem Analisis Klaster Pola Penggunaan AI Mahasiswa</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Program Studi Teknik Informatika — Implementasi K-Means Clustering & Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)

# ==========================================
# 2. LOAD & PREPROCESS DATASET
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data_teknik_informatika.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, 'data.csv')
    
    df = pd.read_csv(data_path)
    
    column_mapping = {
        'Seberapa sering Anda menggunakan AI (ChatGPT, Gemini, Copilot, dll.) untuk menyelesaikan tugas?': 'Frekuensi_AI',
        'Seberapa sering AI membantu Anda dalam Mencari Referensi Tugas?': 'AI_Referensi',
        'Seberapa sering AI membantu Anda dalam Menyusun Kerangka Tugas?': 'AI_Kerangka',
        'Seberapa sering AI membantu Anda dalam  Mengecek grammar/plagiarisme ?': 'AI_Grammar',
        'AI meningkatkan kualitas tugas saya (lebih akurat, rapi, orisinal).': 'Kualitas_Tugas',
        'AI membuat saya lebih efisien (menghemat waktu).': 'Efisiensi',
        'Saya merasa ketergantungan pada AI untuk menyelesaikan tugas.': 'Ketergantungan',
        'Penggunaan AI mengurangi pemahaman mendalam saya terhadap materi.': 'Reduksi_Pemahaman',
        'Rata-rata waktu penyelesaian tugas tanpa AI (jam): _______________': 'Waktu_Tanpa_AI',
        'Rata-rata waktu penyelesaian tugas dengan AI (jam): _______________': 'Waktu_Dengan_AI',
        'Rata-rata nilai tugas sebelum menggunakan AI: _______________': 'Nilai_Sebelum_AI',
        'Rata-rata nilai tugas setelah menggunakan AI: _______________': 'Nilai_Setelah_AI'
    }
    df = df.rename(columns=column_mapping)
    if 'NO' not in df.columns:
        df.insert(0, 'NO', range(1, len(df) + 1))
        
    features = [
        'Frekuensi_AI', 'AI_Referensi', 'AI_Kerangka', 'AI_Grammar',
        'Kualitas_Tugas', 'Efisiensi', 'Ketergantungan', 'Reduksi_Pemahaman',
        'Waktu_Tanpa_AI', 'Waktu_Dengan_AI', 'Nilai_Sebelum_AI', 'Nilai_Setelah_AI'
    ]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[features] = df[features].fillna(df[features].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    return df, features, X_scaled, os.path.basename(data_path)

df, features, X_scaled, filename = load_and_preprocess_data()

# ==========================================
# 3. SIDEBAR KONTROL & PARAMETER
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
    st.title("Panel Parameter")
    st.info(f"📂 Dataset: **{filename}**\n\n👥 Responden: **{len(df)} Mahasiswa**")
    
    st.subheader("⚙️ Parameter K-Means")
    k_clusters = st.slider("Jumlah Klaster (k):", min_value=2, max_value=6, value=3, step=1)
    random_seed = st.number_input("Random State:", value=42, step=1)
    
    st.divider()
    st.caption("Skripsi: Analisis Klaster Pola Penggunaan AI\nTeknik Informatika")

# ==========================================
# 4. MODEL CLUSTERING & PCA COMPUTATION
# ==========================================
kmeans = KMeans(n_clusters=k_clusters, random_state=random_seed, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

label_mapping = {
    0: "Pengguna Kasual / Konservatif (Rendah Adopsi)",
    1: "Pengguna Intensif (Beban Tugas Tinggi & Grammar)",
    2: "Pengguna Strategis & Berprestasi (Efisien & Mandiri)"
}
label_mapping_short = {
    0: "Kasual / Konservatif",
    1: "Intensif (Beban Tinggi)",
    2: "Strategis / Berprestasi"
}
df['Label_Cluster'] = df['Cluster'].map(label_mapping).fillna(df['Cluster'].apply(lambda x: f"Klaster {x}"))
df['Label_Cluster_Short'] = df['Cluster'].map(label_mapping_short).fillna(df['Cluster'].apply(lambda x: f"K-{x}"))

distances = kmeans.transform(X_scaled)
for i in range(k_clusters):
    df[f'Jarak_Cluster_{i}'] = distances[:, i].round(4)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
df['PCA3'] = X_pca[:, 2]
var_ratio = pca.explained_variance_ratio_ * 100
palette = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']

# ==========================================
# 5. TAMPILAN UTAMA BERDASARKAN TAHAPAN (TABS)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Data & Pra-pemrosesan", 
    "2️⃣ Evaluasi Penentuan K", 
    "3️⃣ Klastering & PCA 3D", 
    "4️⃣ Profil Karakteristik Klaster", 
    "5️⃣ Tabel & Ekspor Excel"
])

# ------------------------------------------
# TAB 1: DATA & PRA-PEMROSESAN
# ------------------------------------------
with tab1:
    st.markdown('<div class="step-box"><b>Tahap 1:</b> Pembersihan data, imputation nilai kosong menggunakan median, serta standardisasi fitur (Z-Score) agar tidak ada variabel yang mendominasi akibat perbedaan skala satuan.</div>', unsafe_allow_html=True)
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Total Responden", f"{len(df)} Mahasiswa")
    col_kpi2.metric("Jumlah Fitur Analisis", f"{len(features)} Variabel")
    col_kpi3.metric("Metode Standardisasi", "StandardScaler (Z-Score)")
    
    st.write("### Cuplikan Data Setelah Pembersihan:")
    st.dataframe(df[['NO', 'Jurusan/Fakultas'] + features[:6]].head(10), use_container_width=True)

# ------------------------------------------
# TAB 2: EVALUASI PENENTUAN K
# ------------------------------------------
with tab2:
    st.markdown('<div class="step-box"><b>Tahap 2:</b> Penentuan jumlah klaster optimal (k) menggunakan kombinasi metode <b>Elbow (WCSS)</b> dan <b>Silhouette Score</b>.</div>', unsafe_allow_html=True)
    
    wcss = []
    eval_list = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        if k >= 2:
            s_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            eval_list.append({
                'k': k,
                'WCSS': round(km.inertia_, 2),
                'Silhouette Score': round(s_score, 4),
                'Davies-Bouldin': round(db_score, 4),
                'Calinski-Harabasz': round(ch_score, 2)
            })
    df_eval = pd.DataFrame(eval_list)
    
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.write("#### Grafik Elbow Method (WCSS)")
        fig_elbow, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(range(1, 11), wcss, marker='o', color='#1f77b4', linestyle='--')
        ax1.axvline(x=k_clusters, color='red', linestyle=':', label=f'k Terpilih ({k_clusters})')
        ax1.set_xlabel('Jumlah Klaster (k)')
        ax1.set_ylabel('Inertia (WCSS)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_elbow)
        
    with col_plot2:
        st.write("#### Grafik Silhouette Score")
        fig_sil, ax2 = plt.subplots(figsize=(7, 4.5))
        ax2.plot(df_eval['k'], df_eval['Silhouette Score'], marker='s', color='#d62728')
        ax2.axvline(x=k_clusters, color='red', linestyle=':', label=f'k Terpilih ({k_clusters})')
        ax2.set_xlabel('Jumlah Klaster (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_sil)
        
    st.write("#### Tabel Ringkasan Metrik:")
    st.dataframe(df_eval, use_container_width=True)

# ------------------------------------------
# TAB 3: KLASTERING & PCA 3D
# ------------------------------------------
with tab3:
    st.markdown('<div class="step-box"><b>Tahap 3:</b> Visualisasi hasil klastering menggunakan reduksi dimensi Principal Component Analysis (PCA). Anda dapat memutar grafik 3D di bawah ini dengan kursor mouse untuk demonstrasi interaktif.</div>', unsafe_allow_html=True)
    
    col_pca1, col_pca2 = st.columns([1, 2])
    with col_pca1:
        st.write("#### Proporsi Varians PCA")
        st.write(f"- **PC1:** {var_ratio[0]:.2f}% (Beban Waktu & Nilai)")
        st.write(f"- **PC2:** {var_ratio[1]:.2f}% (Intensitas Adopsi AI)")
        st.write(f"- **PC3:** {var_ratio[2]:.2f}%")
        st.info(f"Total Varians Terjelaskan: **{var_ratio[:3].sum():.2f}%**")
        
        st.write("#### Distribusi Mahasiswa per Klaster:")
        dist_series = df['Label_Cluster_Short'].value_counts()
        st.bar_chart(dist_series)
        
    with col_pca2:
        st.write("#### Scatter Plot 3D Interaktif (Bisa Diputar & Di-zoom):")
        fig_3d = px.scatter_3d(
            df, x='PCA1', y='PCA2', z='PCA3',
            color='Label_Cluster_Short',
            hover_name='NO',
            hover_data=['Frekuensi_AI', 'Nilai_Setelah_AI', 'Waktu_Dengan_AI'],
            color_discrete_sequence=palette[:k_clusters]
        )
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=10),
            scene=dict(
                xaxis_title=f"PC1 ({var_ratio[0]:.1f}%)",
                yaxis_title=f"PC2 ({var_ratio[1]:.1f}%)",
                zaxis_title=f"PC3 ({var_ratio[2]:.1f}%)"
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------
# TAB 4: KARAKTERISTIK & PROFIL KLASTER
# ------------------------------------------
with tab4:
    st.markdown('<div class="step-box"><b>Tahap 4:</b> Mengetahui profil unik dari masing-masing klaster mahasiswa melalui Heatmap rata-rata fitur dan Radar Chart.</div>', unsafe_allow_html=True)
    
    col_prof1, col_prof2 = st.columns([1.2, 1])
    cluster_summary = df.groupby('Label_Cluster_Short')[features].mean()
    
    with col_prof1:
        st.write("#### Heatmap Rata-Rata Fitur:")
        fig_hm, ax_hm = plt.subplots(figsize=(10, 6.5))
        sns.heatmap(cluster_summary.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.7, ax=ax_hm)
        st.pyplot(fig_hm)
        
    with col_prof2:
        st.write("#### Radar Chart Profil:")
        radar_vars = ['Frekuensi_AI', 'Ketergantungan', 'Kualitas_Tugas', 'Nilai_Setelah_AI', 'Waktu_Dengan_AI', 'Reduksi_Pemahaman']
        N = len(radar_vars)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig_radar, ax_r = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for idx, (c_name, row) in enumerate(cluster_summary.iterrows()):
            vals = row[radar_vars].tolist()
            vals += vals[:1]
            ax_r.plot(angles, vals, linewidth=2, label=c_name, color=palette[idx % len(palette)])
            ax_r.fill(angles, vals, alpha=0.15, color=palette[idx % len(palette)])
        
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(radar_vars, fontsize=9)
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)
        st.pyplot(fig_radar)

# ------------------------------------------
# TAB 5: TABEL & EKSPOR EXCEL
# ------------------------------------------
with tab5:
    st.markdown('<div class="step-box"><b>Tahap 5:</b> Penelusuran data hasil prediksi klaster tiap responden beserta jarak Euclidean ke masing-masing centroid. Dilengkapi fitur unduh Excel.</div>', unsafe_allow_html=True)
    
    filter_klaster = st.multiselect(
        "Filter Tampilan Berdasarkan Klaster:",
        options=df['Label_Cluster_Short'].unique(),
        default=df['Label_Cluster_Short'].unique()
    )
    df_filtered = df[df['Label_Cluster_Short'].isin(filter_klaster)]
    
    show_cols = ['NO', 'Label_Cluster_Short'] + [f'Jarak_Cluster_{i}' for i in range(k_clusters)] + ['Frekuensi_AI', 'Nilai_Setelah_AI', 'Waktu_Dengan_AI']
    st.dataframe(df_filtered[show_cols], use_container_width=True)
    
    # Tombol Download Excel In-Memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Hasil_Clustering', index=False)
        cluster_summary.to_excel(writer, sheet_name='Rata_Rata_Fitur')
    
    st.download_button(
        label="📥 Unduh Laporan Lengkap (Excel)",
        data=buffer.getvalue(),
        file_name="hasil_clustering_mahasiswa_AI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
