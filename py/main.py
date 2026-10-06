# Author: Icankkkkk
# ============================
# 1. IMPORT LIBRARY
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# ============================
# 2. LOAD DATA
# ============================
print("[INFO] Memuat data...")
file_path = "data.csv"
df = pd.read_csv(file_path)

print(f"\n[INFO] Total data responden saat ini: {len(df)}")
print("\nNama kolom yang tersedia:")
print(list(df.columns))

# ============================
# 3. PREPROCESSING - Mapping kolom yang benar
# ============================
print("\n[INFO] Pra-pemrosesan data...")

# Mapping berdasarkan nama kolom yang sebenarnya di CSV
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

# Rename kolom
df = df.rename(columns=column_mapping)

# Buat kolom NO (nomor urut)
df.insert(0, 'NO', range(1, len(df) + 1))

# Fitur yang akan digunakan untuk clustering
features = [
    'Frekuensi_AI', 
    'AI_Referensi', 
    'AI_Kerangka', 
    'AI_Grammar',
    'Kualitas_Tugas', 
    'Efisiensi', 
    'Ketergantungan', 
    'Reduksi_Pemahaman',
    'Waktu_Tanpa_AI', 
    'Waktu_Dengan_AI', 
    'Nilai_Sebelum_AI', 
    'Nilai_Setelah_AI'
]

print("\n[INFO] Fitur yang digunakan untuk clustering:")
for f in features:
    print(f"  - {f}")

# Cek apakah semua kolom ada
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    print(f"\n[ERROR] Kolom yang hilang: {missing_cols}")
    print("\nKolom yang tersedia di DataFrame:")
    print(list(df.columns))
    raise KeyError(f"Kolom tidak ditemukan: {missing_cols}")

# Konversi ke numerik, handle missing values
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values dengan median
df[features] = df[features].fillna(df[features].median())

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# 4.1. METODE SILHOUETTE
# ============================
print("\n[INFO] Menghitung Silhouette Score...")
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Untuk k={k}, Silhouette Score: {silhouette_avg:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='red')
plt.title('Silhouette Score Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid()
plt.show()

# ============================
# 4.2. ELBOW METHOD
# ============================
print("\n[INFO] Menentukan jumlah cluster optimal...")
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='blue', linestyle='--')
plt.title('Elbow Method', fontsize=14, pad=20)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# ============================
# 5. CLUSTERING DENGAN K-MEANS
# ============================
optimal_k = 3
print(f"\n[INFO] Menjalankan K-Means dengan {optimal_k} cluster...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Definisikan mapping label
label_mapping = {
    0: "Pengguna Seimbang (Optimal)",
    1: "Pengguna Minimal",
    2: "Pengguna Intensif (Berisiko Ketergantungan)"
}
df['Label_Cluster'] = df['Cluster'].map(label_mapping)

# ============================
# 6. TABEL JARAK & CLUSTER MAHASISWA
# ============================
print("\n[INFO] Menghitung jarak antar centroid...")

distances = kmeans.transform(X_scaled) 
df['Jarak_Cluster_0'] = distances[:, 0]
df['Jarak_Cluster_1'] = distances[:, 1]
df['Jarak_Cluster_2'] = distances[:, 2]

# Menyiapkan tabel untuk diekspor
tabel_eksport = df[['NO', 'Cluster', 'Jarak_Cluster_0', 'Jarak_Cluster_1', 'Jarak_Cluster_2', 'Label_Cluster']].copy()
for col in ['Jarak_Cluster_0', 'Jarak_Cluster_1', 'Jarak_Cluster_2']:
    tabel_eksport[col] = tabel_eksport[col].round(4)

print("\n[INFO] Tabel Mahasiswa dan Cluster (5 Baris Pertama):")
print(tabel_eksport.head().to_string(index=False))

# Simpan tabel lengkap ke Excel
tabel_eksport.to_excel("tabel_cluster_mahasiswa_lengkap.xlsx", index=False)

# ============================
# 7. ANALISIS CLUSTER & VISUALISASI
# ============================
print("\n[INFO] Analisis karakteristik cluster:")
cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Label_Cluster', palette='viridis', s=100)
plt.title('Visualisasi Cluster dengan PCA 2D')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Kategori')
plt.grid()
plt.show()

# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_summary.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
plt.title('Heatmap Rata-Rata Fitur per Cluster', pad=20)
plt.xlabel('Cluster')
plt.ylabel('Fitur')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# PCA 3D (Matplotlib)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
df['PCA3'] = X_pca_3d[:, 2]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                     c=df['Cluster'], cmap='viridis', s=60)

ax.set_title('Visualisasi 3D PCA Clustering', pad=20)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.colorbar(scatter)
plt.show()

# Radar chart
print("\n[INFO] Membuat profil cluster yang lebih informatif (Radar Chart)...")
def plot_radar_chart(cluster_data, features_radar):
    categories = features_radar
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for cluster in cluster_data.index:
        row = cluster_data.loc[cluster]
        values = row.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'{label_mapping.get(cluster, cluster)}')
        ax.fill(angles, values, alpha=0.25)

    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=8)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Profil Cluster (Radar Chart)', pad=20)
    plt.tight_layout()
    plt.show()

selected_features = ['Frekuensi_AI', 'Ketergantungan', 'Kualitas_Tugas', 
                    'Nilai_Setelah_AI', 'Waktu_Dengan_AI', 'Reduksi_Pemahaman']
plot_radar_chart(cluster_summary[selected_features], selected_features)

# Plotly Interaktif 3D
fig_3d = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3',
                    color='Label_Cluster', 
                    hover_name='NO',
                    hover_data={f: True for f in features[:5]},
                    title='Visualisasi 3D Interaktif Cluster Mahasiswa',
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    width=1000, height=800)
fig_3d.update_layout(scene=dict(
                    xaxis_title='PC1 - Intensitas AI',
                    yaxis_title='PC2 - Dampak Akademik',
                    zaxis_title='PC3'),
                  margin=dict(l=0, r=0, b=0, t=30),
                  legend_title="Kategori")
fig_3d.show()

# ============================
# 8. TABEL INTERAKTIF & DISTRIBUSI PERSENTASE
# ============================
print("\n[INFO] Menyiapkan tabel data interaktif dan plot distribusi...")

# Tabel interaktif Plotly - DIPERBAIKI
table_header = dict(
    values=["<b>No</b>", "<b>Cluster</b>", "<b>Jarak ke Cluster 0</b>", "<b>Jarak ke Cluster 1</b>", "<b>Jarak ke Cluster 2</b>", "<b>Deskripsi Cluster</b>"],
    fill_color='paleturquoise',
    align='center',
    font=dict(size=12, color='black')
)

table_cells = dict(
    values=[
        df['NO'],
        df['Cluster'],
        df['Jarak_Cluster_0'].round(4),
        df['Jarak_Cluster_1'].round(4),
        df['Jarak_Cluster_2'].round(4),
        df['Label_Cluster']
    ],
    fill_color='lavender',
    align='center',
    font=dict(size=11),
    height=30
)

fig_table = go.Figure(data=[go.Table(header=table_header, cells=table_cells)])

fig_table.update_layout(
    title="Tabel Interaktif: Seluruh Mahasiswa dan Hasil Clustering",
    width=1000,
    height=800
)
fig_table.show()

# Hitung Persentase
cluster_counts_pct = df['Label_Cluster'].value_counts(normalize=True) * 100
print("\nDistribusi Mahasiswa per Kategori Cluster (%):")
print(cluster_counts_pct.round(2))

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts_pct, labels=cluster_counts_pct.index, autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette("Set2"))
plt.title("Distribusi Pengguna AI Berdasarkan Kategori Cluster", pad=20)
plt.show()

# Barplot
plt.figure(figsize=(10, 6))
colors = sns.color_palette("Set2", n_colors=len(cluster_counts_pct))
bars = plt.bar(cluster_counts_pct.index, cluster_counts_pct.values, color=colors)
plt.title("Persentase Mahasiswa per Kategori Cluster", pad=20)
plt.ylabel("Persentase (%)")
plt.xticks(rotation=15)
plt.ylim(0, 100)
for bar, value in zip(bars, cluster_counts_pct.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f"{value:.1f}%", ha='center', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ============================
# 9. REKAP DISTRIBUSI CLUSTER
# ============================
print("\n[INFO] Rekapitulasi Distribusi Mahasiswa per Cluster:")

cluster_counts_abs = df['Cluster'].value_counts().sort_index()
cluster_percent = (cluster_counts_abs / cluster_counts_abs.sum() * 100).round(2)

cluster_distribution = pd.DataFrame({
    'Jumlah Mahasiswa': cluster_counts_abs,
    'Persentase (%)': cluster_percent,
    'Deskripsi': cluster_counts_abs.index.map(label_mapping)
})

print(cluster_distribution)

# ============================
# 10. SIMPAN HASIL ANALISIS
# ============================
print("\n[INFO] Menyimpan seluruh hasil analisis ke Excel...")
output_file = "hasil_clustering_mahasiswa_AI.xlsx"
df.to_excel(output_file, index=False)

# Tambahkan sheet statistik cluster
with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    cluster_stats = df.groupby('Cluster')[features].agg(['mean', 'std', 'count'])
    cluster_stats.to_excel(writer, sheet_name='Cluster_Statistics')
    
    pca_components = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
    pca_components.to_excel(writer, sheet_name='PCA_Components')
    
    # Simpan juga ringkasan distribusi
    cluster_distribution.to_excel(writer, sheet_name='Distribusi_Cluster')

print(f"\n[INFO] Proses Selesai!")
print(f"- Silhouette Score terbaik: {max(silhouette_scores):.4f} (k={k_range[np.argmax(silhouette_scores)]})")
print(f"- Total data yang diproses: {len(df)}")
print("- Hasil lengkap telah disimpan dalam file Excel: hasil_clustering_mahasiswa_AI.xlsx")
print("- Tabel jarak cluster disimpan di: tabel_cluster_mahasiswa_lengkap.xlsx")