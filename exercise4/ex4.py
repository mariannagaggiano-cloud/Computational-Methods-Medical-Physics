import numpy as np
import matplotlib.pyplot as plt

# ===================== FUNZIONI PRINCIPALI =====================

def euclidean_distance(x, m):
    """Distanza euclidea tra punto x e media m"""
    return np.sqrt(np.sum((x - m) ** 2))

def initialize_means(data, k):
    """Inizializza k medie random"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return np.random.uniform(min_vals, max_vals, (k, data.shape[1]))

def assign_clusters(data, means):
    """Assegna ogni punto al cluster più vicino"""
    clusters = np.zeros(len(data), dtype=int)
    for i, point in enumerate(data):
        distances = [euclidean_distance(point, mean) for mean in means]
        clusters[i] = np.argmin(distances)
    return clusters

def calculate_tv(data, means, clusters, k):
    """Calcola Total Variation"""
    tv = 0
    for c in range(k):
        cluster_points = data[clusters == c]
        for point in cluster_points:
            tv += euclidean_distance(point, means[c]) ** 2
    return tv

def update_means(data, clusters, k):
    """Aggiorna le medie"""
    means = np.zeros((k, data.shape[1]))
    for c in range(k):
        cluster_points = data[clusters == c]
        if len(cluster_points) > 0:
            means[c] = np.mean(cluster_points, axis=0)
    return means

def kmeans(data, k, max_iter=100):
    """Algoritmo k-means completo"""
    means = initialize_means(data, k)
    tv_history = []
    
    for iteration in range(max_iter):
        clusters = assign_clusters(data, means)
        tv = calculate_tv(data, means, clusters, k)
        tv_history.append(tv)
        
        # Convergenza: TV cambia < 1%
        if iteration > 0:
            change = abs(tv_history[-1] - tv_history[-2]) / tv_history[-2] * 100
            if change < 1.0:
                break
        
        means = update_means(data, clusters, k)
    
    return means, clusters, tv_history[0], tv_history[-1]

def silhouette_score(data, clusters, k):
    """Calcola silhouette score medio"""
    if k == 1:
        return 0
    
    scores = []
    for i in range(len(data)):
        cluster_i = clusters[i]
        same_cluster = data[clusters == cluster_i]
        
        if len(same_cluster) <= 1:
            continue
        
        # a_i: distanza media intra-cluster
        a_i = np.mean([np.linalg.norm(data[i] - p) for p in same_cluster 
                       if not np.array_equal(p, data[i])])
        
        # b_i: distanza minima inter-cluster
        b_i = float('inf')
        for c in range(k):
            if c != cluster_i:
                other_cluster = data[clusters == c]
                if len(other_cluster) > 0:
                    dist = np.mean([np.linalg.norm(data[i] - p) for p in other_cluster])
                    b_i = min(b_i, dist)
        
        # s_i
        s_i = (b_i - a_i) / max(a_i, b_i) if b_i != float('inf') else 0
        scores.append(s_i)
    
    return np.mean(scores) if scores else 0

# ===================== MAIN =====================

print("📂 Caricamento dati...")
data = np.loadtxt('s2.txt')
print(f"✓ Caricati {len(data)} punti\n")

# 1. ELBOW METHOD
print("📊 Metodo Elbow...")
k_range = range(2, 21)
tv_values = []

for k in k_range:
    _, _, _, final_tv = kmeans(data, k)
    tv_values.append(final_tv)
    print(f"  k={k}: TV={final_tv:.0f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, tv_values, 'bo-', linewidth=2)
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Total Variation')
plt.title('Elbow Method')
plt.grid(True, alpha=0.3)
plt.savefig('elbow_method.png', dpi=200)
print("✓ Salvato: elbow_method.png\n")

# 2. SILHOUETTE METHOD
print("📊 Metodo Silhouette...")
silhouette_scores = []

for k in k_range:
    means, clusters, _, _ = kmeans(data, k)
    score = silhouette_score(data, clusters, k)
    silhouette_scores.append(score)
    print(f"  k={k}: Score={score:.3f}")

plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2)
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.grid(True, alpha=0.3)
plt.savefig('silhouette_method.png', dpi=200)
print("✓ Salvato: silhouette_method.png\n")

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"🎯 K ottimale (Silhouette): {optimal_k}\n")

# 3. CLUSTERING FINALE con k=15
print("🔧 Clustering finale con k=15...")
means, clusters, init_tv, final_tv = kmeans(data, 15)
print(f"  TV iniziale: {init_tv:.0f}")
print(f"  TV finale: {final_tv:.0f}")

# 4. REPORT
print("\n" + "="*50)
print("📋 REPORT FINALE")
print("="*50)
print(f"Numero cluster: k=15")
print(f"TV iniziale: {init_tv:.2f}")
print(f"TV finale: {final_tv:.2f}")
print(f"Riduzione: {(init_tv-final_tv)/init_tv*100:.1f}%")

cluster_sizes = [np.sum(clusters == c) for c in range(15)]
print(f"\nCluster più piccolo: {min(cluster_sizes)} punti")
print(f"Cluster più grande: {max(cluster_sizes)} punti")
print("\nDimensioni cluster:")
for i, size in enumerate(cluster_sizes):
    print(f"  Cluster {i+1:2d}: {size:4d} punti")

# 5. VISUALIZZAZIONE
print("\n🎨 Generazione figura finale...")
plt.figure(figsize=(12, 10))
colors = plt.cm.rainbow(np.linspace(0, 1, 15))

for c in range(15):
    cluster_points = data[clusters == c]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=[colors[c]], label=f'C{c+1}', alpha=0.6, s=15)

plt.scatter(means[:, 0], means[:, 1], c='black', marker='X', 
           s=200, edgecolors='white', linewidths=2, label='Centroids')

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'K-means (k=15) - TV init: {init_tv:.0f}, TV final: {final_tv:.0f}')
plt.legend(bbox_to_anchor=(1.05, 1), ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_clustering.png', dpi=200)
print("✓ Salvato: final_clustering.png")

print("\n✅ COMPLETATO!\n")
print("File generati:")
print("  • elbow_method.png")
print("  • silhouette_method.png")
print("  • final_clustering.png")