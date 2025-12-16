import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class KMeans:
    def __init__(self, k, max_iter=100, threshold=0.01):
        """
        Inizializza l'algoritmo k-means
        k: numero di cluster
        max_iter: numero massimo di iterazioni
        threshold: soglia per la convergenza (variazione TV in %)
        """
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.means = None
        self.clusters = None
        self.tv_history = []
        
    def euclidean_distance(self, x, m):
        """Calcola la distanza euclidea tra punto x e media m"""
        return np.sqrt(np.sum((x - m) ** 2))
    
    def initialize_means(self, data):
        """Inizializza k medie random nello spazio dei dati"""
        n_features = data.shape[1]
        # Sample random punti nell'intervallo dei dati
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.means = np.random.uniform(min_vals, max_vals, (self.k, n_features))
    
    def assign_clusters(self, data):
        """Assegna ogni punto al cluster con la media più vicina"""
        n_samples = data.shape[0]
        clusters = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = [self.euclidean_distance(data[i], mean) for mean in self.means]
            clusters[i] = np.argmin(distances)
        
        return clusters
    
    def calculate_total_variation(self, data, clusters):
        """Calcola la variazione totale (TV)"""
        tv = 0
        for cluster_idx in range(self.k):
            cluster_points = data[clusters == cluster_idx]
            if len(cluster_points) > 0:
                for point in cluster_points:
                    tv += self.euclidean_distance(point, self.means[cluster_idx]) ** 2
        return tv
    
    def update_means(self, data, clusters):
        """Aggiorna le medie basandosi sui punti di ogni cluster"""
        new_means = np.zeros((self.k, data.shape[1]))
        for cluster_idx in range(self.k):
            cluster_points = data[clusters == cluster_idx]
            if len(cluster_points) > 0:
                new_means[cluster_idx] = np.mean(cluster_points, axis=0)
            else:
                # Se un cluster è vuoto, reinizializza random
                new_means[cluster_idx] = self.means[cluster_idx]
        return new_means
    
    def fit(self, data):
        """Esegue l'algoritmo k-means"""
        # Step 2: Inizializza medie random
        self.initialize_means(data)
        
        initial_tv = None
        
        for iteration in range(self.max_iter):
            # Step 3-4: Assegna punti ai cluster
            self.clusters = self.assign_clusters(data)
            
            # Step 5: Calcola TV
            tv = self.calculate_total_variation(data, self.clusters)
            self.tv_history.append(tv)
            
            if initial_tv is None:
                initial_tv = tv
            
            # Check convergenza
            if iteration > 0:
                tv_change = abs(self.tv_history[-1] - self.tv_history[-2])
                tv_change_percent = (tv_change / self.tv_history[-2]) * 100
                if tv_change_percent < self.threshold:
                    print(f"Convergenza raggiunta all'iterazione {iteration}")
                    break
            
            # Step 6: Aggiorna medie
            self.means = self.update_means(data, self.clusters)
        
        return initial_tv, self.tv_history[-1]


def elbow_method(data, k_range, n_runs=5):
    """Metodo elbow per determinare il numero ottimale di cluster"""
    tv_values = []
    
    for k in k_range:
        print(f"Testing k={k}...")
        best_tv = float('inf')
        
        # Esegui più volte per ogni k e prendi il miglior risultato
        for run in range(n_runs):
            kmeans = KMeans(k=k, max_iter=100, threshold=0.01)
            _, final_tv = kmeans.fit(data)
            if final_tv < best_tv:
                best_tv = final_tv
        
        tv_values.append(best_tv)
    
    return tv_values


def silhouette_method(data, k_range, n_runs=5):
    """Metodo silhouette per determinare il numero ottimale di cluster"""
    silhouette_scores = []
    
    for k in k_range:
        if k == 1:
            silhouette_scores.append(0)
            continue
            
        print(f"Testing k={k} (Silhouette)...")
        best_score = -1
        best_kmeans = None
        
        # Esegui più volte per ogni k
        for run in range(n_runs):
            kmeans = KMeans(k=k, max_iter=100, threshold=0.01)
            kmeans.fit(data)
            score = calculate_silhouette_score(data, kmeans.clusters, k)
            if score > best_score:
                best_score = score
                best_kmeans = kmeans
        
        silhouette_scores.append(best_score)
    
    return silhouette_scores


def calculate_silhouette_score(data, clusters, k):
    """Calcola lo score silhouette medio"""
    n_samples = data.shape[0]
    silhouette_values = []
    
    for i in range(n_samples):
        cluster_i = clusters[i]
        
        # Calcola a_i (similarità intra-cluster)
        same_cluster_points = data[clusters == cluster_i]
        if len(same_cluster_points) <= 1:
            silhouette_values.append(0)
            continue
        
        a_i = np.mean([np.linalg.norm(data[i] - point) 
                       for point in same_cluster_points if not np.array_equal(point, data[i])])
        
        # Calcola b_i (dissimilarità inter-cluster)
        b_i = float('inf')
        for cluster_j in range(k):
            if cluster_j != cluster_i:
                other_cluster_points = data[clusters == cluster_j]
                if len(other_cluster_points) > 0:
                    mean_dist = np.mean([np.linalg.norm(data[i] - point) 
                                        for point in other_cluster_points])
                    b_i = min(b_i, mean_dist)
        
        # Calcola s_i
        s_i = (b_i - a_i) / max(a_i, b_i)
        silhouette_values.append(s_i)
    
    return np.mean(silhouette_values)


def plot_results(data, kmeans, initial_tv, final_tv):
    """Visualizza i risultati finali del clustering"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crea colormap
    colors = plt.cm.rainbow(np.linspace(0, 1, kmeans.k))
    
    # Plotta ogni cluster
    for cluster_idx in range(kmeans.k):
        cluster_points = data[kmeans.clusters == cluster_idx]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  c=[colors[cluster_idx]], label=f'Cluster {cluster_idx+1}',
                  alpha=0.6, s=20)
    
    # Plotta le medie
    ax.scatter(kmeans.means[:, 0], kmeans.means[:, 1], 
              c='black', marker='X', s=200, edgecolors='white', linewidths=2,
              label='Centroids', zorder=5)
    
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'K-means Clustering Results (k={kmeans.k})\n'
                f'Initial TV: {initial_tv:.2f}, Final TV: {final_tv:.2f}', 
                fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ===================== MAIN =====================

# Carica i dati
print("Caricamento dati s2.txt...")
data = np.loadtxt('s2.txt')
print(f"Dati caricati: {data.shape[0]} punti con {data.shape[1]} features")

# 1. METODO ELBOW
print("\n=== METODO ELBOW ===")
k_range = range(2, 21)
tv_values = elbow_method(data, k_range, n_runs=5)

# Plot elbow
plt.figure(figsize=(10, 6))
plt.plot(k_range, tv_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Numero di cluster (k)', fontsize=12)
plt.ylabel('Total Variation', fontsize=12)
plt.title('Elbow Method', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
print("Grafico elbow salvato come 'elbow_method.png'")

# 2. METODO SILHOUETTE
print("\n=== METODO SILHOUETTE ===")
silhouette_scores = silhouette_method(data, k_range, n_runs=5)

# Plot silhouette
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Numero di cluster (k)', fontsize=12)
plt.ylabel('Silhouette Score medio', fontsize=12)
plt.title('Silhouette Method', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('silhouette_method.png', dpi=300, bbox_inches='tight')
print("Grafico silhouette salvato come 'silhouette_method.png'")

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nNumero ottimale di cluster (Silhouette): k={optimal_k}")

# 3. CLUSTERING FINALE CON K OTTIMALE (o k=15 come nel documento)
print(f"\n=== CLUSTERING FINALE CON k=15 ===")
best_kmeans = None
best_final_tv = float('inf')
best_initial_tv = None

# Esegui 10 volte e prendi il miglior risultato
for run in range(10):
    print(f"Run {run+1}/10...")
    kmeans = KMeans(k=15, max_iter=100, threshold=0.01)
    initial_tv, final_tv = kmeans.fit(data)
    
    if final_tv < best_final_tv:
        best_final_tv = final_tv
        best_initial_tv = initial_tv
        best_kmeans = kmeans

# 4. REPORT DEI RISULTATI
print("\n" + "="*60)
print("REPORT FINALE")
print("="*60)
print(f"Numero di cluster: k={best_kmeans.k}")
print(f"Total Variation iniziale: {best_initial_tv:.2f}")
print(f"Total Variation finale: {best_final_tv:.2f}")
print(f"Riduzione TV: {((best_initial_tv - best_final_tv) / best_initial_tv * 100):.2f}%")

# Dimensioni dei cluster
cluster_sizes = [np.sum(best_kmeans.clusters == i) for i in range(best_kmeans.k)]
print(f"\nCluster più piccolo: {min(cluster_sizes)} punti")
print(f"Cluster più grande: {max(cluster_sizes)} punti")
print(f"\nDistribuzione dimensioni cluster:")
for i, size in enumerate(cluster_sizes):
    print(f"  Cluster {i+1}: {size} punti")

# 5. VISUALIZZAZIONE FINALE
print("\nGenerazione figura finale...")
fig = plot_results(data, best_kmeans, best_initial_tv, best_final_tv)
plt.savefig('final_clustering.png', dpi=300, bbox_inches='tight')
print("Figura finale salvata come 'final_clustering.png'")

plt.show()

print("\n✓ Esercizio completato! File generati:")
print("  - elbow_method.png")
print("  - silhouette_method.png")
print("  - final_clustering.png")