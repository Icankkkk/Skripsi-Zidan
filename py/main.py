# Author: Icankkkkk
# Refactored & Enhanced for Skripsi Zidan (Analisis Klaster Mahasiswa Teknik Informatika)
# ==============================================================================
# Pipeline Analisis Pola Penggunaan AI dengan K-Means Clustering & PCA
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go

# Set backend matplotlib agar tidak membekukan eksekusi terminal
# Ubah SHOW_PLOTS = True jika ingin menampilkan jendela popup grafik satu per satu
SHOW_PLOTS = False
if not SHOW_PLOTS:
    matplotlib.use('Agg')

# ============================
# 1. SETUP DIREKTORI & PATH
# ============================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
if os.path.exists(os.path.join(script_dir, "data_teknik_informatika.csv")):
    base_dir = script_dir
elif os.path.exists(os.path.join(script_dir, "py", "data_teknik_informatika.csv")):
    base_dir = os.path.join(script_dir, "py")
else:
    base_dir = os.getcwd()

output_dir = os.path.join(base_dir, 'output')
images_dir = os.path.join(output_dir, 'Images')
os.makedirs(images_dir, exist_ok=True)

# Prioritaskan dataset khusus Teknik Informatika
data_path = os.path.join(base_dir, 'data_teknik_informatika.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(base_dir, 'data.csv')

# ============================
# 2. LOAD DATA
# ============================
print("=" * 60)
print(f"[INFO] Memuat dataset dari: {data_path}")
print("=" * 60)

df = pd.read_csv(data_path)
print(f"[INFO] Total data responden: {len(df)}")
print(f"[INFO] Program Studi: {df['Jurusan/Fakultas'].unique() if 'Jurusan/Fakultas' in df.columns else 'N/A'}")

# ============================
# 3. PREPROCESSING & MAPPING
# ============================
print("\n[INFO] Melakukan pra-pemrosesan data...")

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

print(f"[INFO] {len(features)} fitur yang dianalisis untuk clustering:")
for f in features:
    print(f"  - {f}")

# Validasi & konversi numerik
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Imputasi nilai kosong dengan median
df[features] = df[features].fillna(df[features].median())

# Standardisasi fitur
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# 4. EVALUASI JUMLAH KLASTER (K)
# ============================
print("\n[INFO] Menghitung metrik evaluasi klaster (WCSS, Silhouette, Davies-Bouldin, Calinski-Harabasz)...")
wcss = []
eval_metrics = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    if k >= 2:
        s_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
        eval_metrics.append({
            'k': k,
            'WCSS (Inertia)': round(km.inertia_, 2),
            'Silhouette Score': round(s_score, 4),
            'Davies-Bouldin Index': round(db_score, 4),
            'Calinski-Harabasz Index': round(ch_score, 2)
        })

df_eval = pd.DataFrame(eval_metrics)
print("\n=== TABEL EVALUASI JUMLAH KLASTER (k) ===")
print(df_eval.to_string(index=False))

# 4.1 Plot Silhouette Score Analysis
plt.figure(figsize=(10, 5))
plt.plot(df_eval['k'], df_eval['Silhouette Score'], marker='o', color='#d62728', linewidth=2)
plt.title('Silhouette Score Analysis (Evaluasi Pemisahan Klaster)', fontsize=13, pad=15)
plt.xlabel('Jumlah Klaster (k)', fontsize=11)
plt.ylabel('Silhouette Score', fontsize=11)
plt.xticks(df_eval['k'])
plt.grid(True, linestyle='--', alpha=0.6)
sil_img_path = os.path.join(images_dir, '01_silhouette_score_analysis.png')
plt.savefig(sil_img_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"[SAVED] {sil_img_path}")

# 4.2 Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='#1f77b4', linestyle='--', linewidth=2)
plt.title('Elbow Method (Penentuan Jumlah Klaster Optimal via WCSS)', fontsize=13, pad=15)
plt.xlabel('Jumlah Klaster (k)', fontsize=11)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=11)
plt.xticks(range(1, 11))
plt.grid(True, linestyle='--', alpha=0.6)
elbow_img_path = os.path.join(images_dir, '02_elbow_method_wcss.png')
plt.savefig(elbow_img_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"[SAVED] {elbow_img_path}")

# ============================
# 5. CLUSTERING K-MEANS (k = 3)
# ============================
optimal_k = 3
print(f"\n[INFO] Menjalankan K-Means dengan k={optimal_k} klaster (random_state=42)...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Pelabelan akurat berdasarkan karakteristik data centroid empiris:
# Cluster 0: Frekuensi AI rendah (2.34), namun kekhawatiran ketergantungan tertinggi (3.52)
# Cluster 1: Frekuensi AI tinggi (4.15), beban tugas terberat (6.88 jam), fokus efisiensi & grammar (3.68)
# Cluster 2: Frekuensi AI tertinggi (4.20), referensi & kerangka tinggi, tugas cepat (2.2 jam), nilai tertinggi (4.29)
label_mapping = {
    0: "Pengguna Kasual / Konservatif (Rendah Adopsi, Khawatir Ketergantungan)",
    1: "Pengguna Intensif (Beban Tugas Tinggi & Penunjang Bahasa)",
    2: "Pengguna Strategis & Berprestasi Tinggi (Efisien & Mandiri)"
}
label_mapping_short = {
    0: "Kasual / Konservatif",
    1: "Intensif (Beban Tinggi)",
    2: "Strategis / Berprestasi"
}
df['Label_Cluster'] = df['Cluster'].map(label_mapping)
df['Label_Cluster_Short'] = df['Cluster'].map(label_mapping_short)

# Jarak Euclidean responden ke masing-masing centroid
distances = kmeans.transform(X_scaled)
for i in range(optimal_k):
    df[f'Jarak_Cluster_{i}'] = distances[:, i].round(4)

# ============================
# 6. PCA (PRINCIPAL COMPONENT ANALYSIS)
# ============================
print("\n[INFO] Melakukan reduksi dimensi dengan PCA...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
df['PCA3'] = X_pca[:, 2]

var_ratio = pca.explained_variance_ratio_ * 100
print(f"[INFO] Proporsi Varians Terjelaskan (Explained Variance):")
print(f"  - PC1: {var_ratio[0]:.2f}% (Beban Waktu Tugas & Performa Akademik)")
print(f"  - PC2: {var_ratio[1]:.2f}% (Frekuensi & Intensitas Penggunaan AI)")
print(f"  - PC3: {var_ratio[2]:.2f}%")
print(f"  - Total Varians (PC1+PC2+PC3): {var_ratio.sum():.2f}%")

# ============================
# 7. VISUALISASI KLASTER
# ============================
print("\n[INFO] Menghasilkan grafik visualisasi beresolusi tinggi (300 DPI)...")
palette = ['#e74c3c', '#3498db', '#2ecc71']

# 7.1 PCA 2D Scatter Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df, x='PCA1', y='PCA2', hue='Label_Cluster_Short',
    palette=palette, s=90, alpha=0.9, edgecolor='black', linewidth=0.5
)
plt.title('Visualisasi Klaster Mahasiswa dengan PCA 2D', fontsize=14, pad=15)
plt.xlabel(f'PC1: Durasi Pengerjaan & Performa Akademik ({var_ratio[0]:.1f}%)', fontsize=11)
plt.ylabel(f'PC2: Intensitas Pemanfaatan AI ({var_ratio[1]:.1f}%)', fontsize=11)
plt.legend(title='Kategori Klaster', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
pca2d_path = os.path.join(images_dir, '03_pca_2d_clusters.png')
plt.savefig(pca2d_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"[SAVED] {pca2d_path}")

# 7.2 Heatmap Rata-Rata Fitur per Klaster
cluster_summary = df.groupby('Label_Cluster_Short')[features].mean()
plt.figure(figsize=(12, 7))
sns.heatmap(
    cluster_summary.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.7,
    cbar_kws={'label': 'Nilai Rata-rata'}
)
plt.title('Heatmap Karakteristik Rata-Rata Fitur per Klaster', fontsize=14, pad=15)
plt.xlabel('Kategori Klaster', fontsize=11)
plt.ylabel('Fitur Analisis', fontsize=11)
plt.xticks(rotation=0, ha='center', fontsize=10)
plt.yticks(fontsize=10)
heatmap_path = os.path.join(images_dir, '04_heatmap_cluster_features.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"[SAVED] {heatmap_path}")

# 7.3 PCA 3D Matplotlib
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
for c_id, name in label_mapping_short.items():
    sub = df[df['Cluster'] == c_id]
    ax.scatter(
        sub['PCA1'], sub['PCA2'], sub['PCA3'], 
        c=palette[c_id], label=name, s=60, alpha=0.85, edgecolor='k', linewidth=0.3
    )
ax.set_title('Visualisasi 3D PCA Clustering Mahasiswa', fontsize=13, pad=15)
ax.set_xlabel(f'PC1 ({var_ratio[0]:.1f}%)', fontsize=10)
ax.set_ylabel(f'PC2 ({var_ratio[1]:.1f}%)', fontsize=10)
ax.set_zlabel(f'PC3 ({var_ratio[2]:.1f}%)', fontsize=10)
ax.legend(title='Klaster', loc='upper left')
pca3d_path = os.path.join(images_dir, '05_pca_3d_matplotlib.png')
plt.savefig(pca3d_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"[SAVED] {pca3d_path}")

# 7.4 Radar Chart Profil Klaster
def plot_radar_chart(data_mean, selected_vars, save_dest):
    categories = selected_vars
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for idx, (cluster_name, row) in enumerate(data_mean.iterrows()):
        values = row[selected_vars].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cluster_name, color=palette[idx])
        ax.fill(angles, values, alpha=0.2, color=palette[idx])

    plt.xticks(angles[:-1], categories, color='black', size=10)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=8)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Klaster')
    plt.title('Profil Karakteristik Klaster (Radar Chart)', fontsize=14, pad=25)
    plt.savefig(save_dest, dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

selected_radar = ['Frekuensi_AI', 'Ketergantungan', 'Kualitas_Tugas', 'Nilai_Setelah_AI', 'Waktu_Dengan_AI', 'Reduksi_Pemahaman']
radar_path = os.path.join(images_dir, '06_radar_chart_profiles.png')
plot_radar_chart(cluster_summary, selected_radar, radar_path)
print(f"[SAVED] {radar_path}")

# 7.5 Distribusi Pie Chart
cluster_dist = df['Label_Cluster_Short'].value_counts()
cluster_pct = df['Label_Cluster_Short'].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 8))
plt.pie(cluster_pct, labels=cluster_pct.index, autopct='%1.1f%%', startangle=140, colors=palette)
plt.title('Distribusi Mahasiswa Berdasarkan Kategori Klaster', fontsize=13, pad=15)
pie_path = os.path.join(images_dir, '07_pie_chart_distribution.png')
plt.savefig(pie_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"[SAVED] {pie_path}")

# 7.6 Distribusi Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(cluster_pct.index, cluster_pct.values, color=palette, edgecolor='black', linewidth=0.5)
plt.title('Persentase Mahasiswa per Kategori Klaster', fontsize=13, pad=15)
plt.ylabel('Persentase (%)', fontsize=11)
plt.ylim(0, max(cluster_pct.values) + 15)
for bar, value in zip(bars, cluster_pct.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{value:.1f}%", ha='center', fontsize=10, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
bar_path = os.path.join(images_dir, '08_bar_chart_distribution.png')
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
if SHOW_PLOTS:
    plt.show()
plt.close()
print(f"[SAVED] {bar_path}")

# ============================
# 8. VISUALISASI INTERAKTIF PLOTLY
# ============================
print("\n[INFO] Menghasilkan grafik interaktif Plotly (HTML)...")

# 8.1 3D Scatter Interaktif
fig_3d = px.scatter_3d(
    df, x='PCA1', y='PCA2', z='PCA3',
    color='Label_Cluster_Short',
    hover_name='NO',
    hover_data={'Jurusan/Fakultas': True, 'Frekuensi_AI': True, 'Nilai_Setelah_AI': True, 'Waktu_Dengan_AI': True},
    title='Visualisasi 3D Interaktif Klaster Mahasiswa Teknik Informatika',
    color_discrete_sequence=palette
)
fig_3d.update_layout(
    scene=dict(
        xaxis_title=f'PC1: Durasi & Nilai ({var_ratio[0]:.1f}%)',
        yaxis_title=f'PC2: Intensitas AI ({var_ratio[1]:.1f}%)',
        zaxis_title=f'PC3 ({var_ratio[2]:.1f}%)'
    ),
    margin=dict(l=0, r=0, b=0, t=35),
    legend_title="Kategori Klaster"
)
html_3d_path = os.path.join(images_dir, 'interactive_pca_3d.html')
fig_3d.write_html(html_3d_path)
print(f"[SAVED] {html_3d_path}")

# 8.2 Tabel Interaktif Plotly
table_header = dict(
    values=["<b>No</b>", "<b>Klaster</b>", "<b>Jarak Klaster 0</b>", "<b>Jarak Klaster 1</b>", "<b>Jarak Klaster 2</b>", "<b>Kategori Klaster</b>"],
    fill_color='paleturquoise',
    align='center',
    font=dict(size=12, color='black')
)
table_cells = dict(
    values=[
        df['NO'],
        df['Cluster'],
        df['Jarak_Cluster_0'],
        df['Jarak_Cluster_1'],
        df['Jarak_Cluster_2'],
        df['Label_Cluster']
    ],
    fill_color='lavender',
    align='center',
    font=dict(size=11),
    height=28
)
fig_table = go.Figure(data=[go.Table(header=table_header, cells=table_cells)])
fig_table.update_layout(title="Tabel Interaktif Mahasiswa dan Hasil Clustering", width=1100, height=800)
html_table_path = os.path.join(images_dir, 'interactive_table.html')
fig_table.write_html(html_table_path)
print(f"[SAVED] {html_table_path}")

# ============================
# 9. SIMPAN OUTPUT KE EXCEL
# ============================
print("\n[INFO] Menyimpan seluruh hasil analisis ke format Excel...")

table_excel_path = os.path.join(output_dir, 'tabel_cluster_mahasiswa_lengkap.xlsx')
export_dist = df[['NO', 'Cluster', 'Jarak_Cluster_0', 'Jarak_Cluster_1', 'Jarak_Cluster_2', 'Label_Cluster']].copy()
export_dist.to_excel(table_excel_path, index=False)
print(f"[SAVED] {table_excel_path}")

output_excel_path = os.path.join(output_dir, 'hasil_clustering_mahasiswa_AI.xlsx')
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    # Sheet 1: Data Lengkap Responden
    df.to_excel(writer, sheet_name='Data_Mahasiswa_Lengkap', index=False)
    
    # Sheet 2: Statistik Klaster (Mean, Std, Min, Max, Count)
    stats = df.groupby('Label_Cluster')[features].agg(['mean', 'std', 'min', 'max', 'count'])
    stats.to_excel(writer, sheet_name='Statistik_Klaster')
    
    # Sheet 3: Distribusi Klaster
    dist_df = pd.DataFrame({
        'Cluster ID': [0, 1, 2],
        'Kategori Klaster': [label_mapping[0], label_mapping[1], label_mapping[2]],
        'Jumlah Mahasiswa': [cluster_dist.get(label_mapping_short[0], 0), cluster_dist.get(label_mapping_short[1], 0), cluster_dist.get(label_mapping_short[2], 0)],
        'Persentase (%)': [cluster_pct.get(label_mapping_short[0], 0), cluster_pct.get(label_mapping_short[1], 0), cluster_pct.get(label_mapping_short[2], 0)]
    })
    dist_df.to_excel(writer, sheet_name='Distribusi_Klaster', index=False)
    
    # Sheet 4: Komponen PCA (Loadings & Explained Variance)
    pca_loadings = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2', 'PC3'])
    pca_loadings.loc['Explained_Variance_Ratio (%)'] = [var_ratio[0], var_ratio[1], var_ratio[2]] + [np.nan]*(len(features)-3)
    pca_loadings.to_excel(writer, sheet_name='PCA_Components')
    
    # Sheet 5: Evaluasi Metrik Nilai K
    df_eval.to_excel(writer, sheet_name='Evaluasi_Metrik_K', index=False)

print(f"[SAVED] {output_excel_path}")

print("\n" + "=" * 60)
print("[SUCCESS] Seluruh proses clustering, visualisasi, dan ekspor selesai!")
print(f"- Dataset yang digunakan: {os.path.basename(data_path)}")
print(f"- Total responden: {len(df)}")
print(f"- Jumlah klaster optimal (k): {optimal_k}")
print(f"- Gambar grafik tersimpan di: {images_dir}")
print(f"- File Excel utama tersimpan di: {output_excel_path}")
print("=" * 60)
