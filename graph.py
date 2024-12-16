import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv

# Modeli yükleme
data = np.load('model.npz')
W1 = data['W1']
b1 = data['b1']
W2 = data['W2']
b2 = data['b2']

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(x1, x2):
    # Girdi normalizasyonu: x/9
    inp = np.array([[x1/9.0, x2/9.0]])
    Z1_ = np.dot(inp, W1) + b1
    A1_ = sigmoid(Z1_)
    Z2_ = np.dot(A1_, W2) + b2
    # Çıkış geri ölçekleme: *81
    return Z2_[0,0]*81

# Eğitimde kullanılan veri seti
odd_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Eğitim setini tekrar oluşturalım
pairs = [(x, y) for x in odd_numbers for y in odd_numbers]
X_raw = np.array([[p[0], p[1]] for p in pairs], dtype=float)
y_raw = np.array([p[0]*p[1] for p in pairs], dtype=float)

# Tüm eğitim örneklerinde model tahmini
y_pred = np.array([predict(p[0], p[1]) for p in pairs])
y_true = y_raw

### 1. Gerçek Değer vs Tahmin Scatter Grafiği ###
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.7)
plt.plot([0,81],[0,81],'r--', label='y=x')
plt.xlabel('Gerçek Değer')
plt.ylabel('Tahmin Değer')
plt.title('Gerçek vs Tahmin')
plt.grid(True)
plt.legend()
plt.savefig('gercek_vs_tahmin.png', dpi=300)

### 2. Hata Dağılımı Histogramı ###
plt.figure(figsize=(8,4))
errors = y_true - y_pred
plt.hist(errors, bins=10, edgecolor='black')
plt.xlabel('Hata (Gerçek - Tahmin)')
plt.ylabel('Frekans')
plt.title('Hata Dağılımı')
plt.grid(True)
plt.savefig('hata_histogrami.png', dpi=300)

### 3. Tahmin Isı Haritası (Heatmap) ###
plt.figure(figsize=(6,5))
pred_matrix = np.zeros((len(odd_numbers), len(odd_numbers)))
for i, a in enumerate(odd_numbers):
    for j, b in enumerate(odd_numbers):
        pred_matrix[i, j] = predict(a, b)

sns.heatmap(pred_matrix, annot=True, xticklabels=odd_numbers, yticklabels=odd_numbers, cmap="viridis", fmt=".1f")
plt.xlabel('İkinci Sayı')
plt.ylabel('Birinci Sayı')
plt.title('Model Tahmin Isı Haritası')
plt.savefig('isi_haritasi.png', dpi=300)

### 4. 3D Yüzey Grafiği (Surface Plot) ###
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
X_vals = odd_numbers
Y_vals = odd_numbers
X_mesh, Y_mesh = np.meshgrid(X_vals, Y_vals)
Z_pred = np.zeros_like(X_mesh, dtype=float)

for i in range(X_mesh.shape[0]):
    for j in range(X_mesh.shape[1]):
        Z_pred[i,j] = predict(X_mesh[i,j], Y_mesh[i,j])

surface = ax.plot_surface(X_mesh, Y_mesh, Z_pred, cmap=cm.viridis)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Tahmin')
ax.set_title('Model Tahminleri 3D Yüzey Grafiği')
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.savefig('3D_yuzey_grafigi.png', dpi=300)

print("Grafikler dosya olarak kaydedildi.")

epochs = []
costs = []

# cost_log.csv dosyasından verileri oku
with open('cost_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Başlık satırını atla
    for row in reader:
        e = int(row[0])
        c = float(row[1])
        epochs.append(e)
        costs.append(c)

# Epoch vs Cost grafiği
plt.figure(figsize=(10,6))
plt.plot(epochs, costs, label='Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Epoch vs Cost Eğrisi')
plt.grid(True)
plt.legend()
plt.savefig('epoch_vs_cost.png', dpi=300)

print("Epoch vs Cost grafiği kaydedildi. Şimdi ekranda gösteriliyor...")

# Tüm figürleri sırayla göster
plt.show()
