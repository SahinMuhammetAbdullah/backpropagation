import numpy as np
import csv

# CSV dosyasını oluşturup başlık satırını yazalım
with open('cost_log.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Cost'])
# Tek sayılar
odd_numbers = np.array([1, 3, 5, 7, 9])

# Tüm çiftleri oluştur (Cartesian product)
pairs = [(x, y) for x in odd_numbers for y in odd_numbers]

# Girdi ve çıktıların hazırlanması
X_raw = np.array([[p[0], p[1]] for p in pairs], dtype=float)
y_raw = np.array([p[0]*p[1] for p in pairs], dtype=float)

# Normalizasyon: 
# Girdiler 1-9 arası => X'i /9 ile ölçekle
# Çıktılar 1-81 arası (max: 81) => y'yi /81 ile ölçekle
X = X_raw / 9.0
y = y_raw / 81.0
y = y.reshape(-1, 1)

# Ağırlıkların rastgele başlatılması
# Ağ Mimarisi: 2 giriş -> 8 nöronlu gizli katman -> 1 çıkış
np.random.seed(42)
W1 = np.random.randn(2, 8) * 0.1
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.1
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(a):
    return a*(1-a)

learning_rate = 0.9 # Öğrenme katsayısı. Testler sonucu optimum değer "0.27"
epochs = 200000 # İtetasyon sayısı. Testler sonucu optimum değer "200000"
lambda_reg = 0.0001  # Düzenlileştirme parametresi. Testler sonucu optimum değer "0.0001"

m = X.shape[0]  # Örnek sayısı

for i in range(epochs):
    # İleri besleme (Forward Pass)
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2  # Çıkış katmanı lineer
    
    # Hata hesabı (MSE)
    error = A2 - y
    # L2 düzenlileştirme terimi: (lambda_reg/(2*m)) * (||W1||^2 + ||W2||^2)
    reg_term = (lambda_reg/(2*m)) * (np.sum(W1**2) + np.sum(W2**2))
    cost = np.mean(error**2) + reg_term
    
    # Geri yayılım (Backpropagation)
    dZ2 = error  
    # dW2'ye L2 düzenlileştirme ekle: (lambda_reg/m)*W2
    dW2 = (np.dot(A1.T, dZ2)/m) + (lambda_reg/m)*W2
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    # dW1'e L2 düzenlileştirme ekle: (lambda_reg/m)*W1
    dW1 = (np.dot(X.T, dZ1)/m) + (lambda_reg/m)*W1
    db1 = np.mean(dZ1, axis=0, keepdims=True)
    
    # Ağırlık güncelleme
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    # Her 50000 adımda bir hata yazdır
    if i % 50000 == 0:
        print(f"Epoch {i}, Cost: {cost:.6f}")
    
    with open('cost_log.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([i, cost])

# Eğitim tamamlandıktan sonra ağırlıkları ekrana yazdır
print("\nEğitim tamamlandı. Model Ağırlıkları:")
print("W1:\n", W1)
print("b1:\n", b1)
print("W2:\n", W2)
print("b2:\n", b2)

def predict(x1, x2):
    inp = np.array([[x1/9.0, x2/9.0]])
    Z1_ = np.dot(inp, W1) + b1
    A1_ = sigmoid(Z1_)
    Z2_ = np.dot(A1_, W2) + b2
    return Z2_[0,0]*81

# Sabit test seti
test_pairs = [(3,5), (7,9), (1,9), (5,5), (3,3)]
print("\nÖnceden belirlenen test örnekleri:")
for (a,b) in test_pairs:
    pred = predict(a,b)
    print(f"{a} * {b} = {a*b}, Model Tahmini: {pred:.2f}")


print("\nKullanıcı testi (Programdan çıkmak için 'q' veya 'Q' girin):")
while True:
    user_input = input("İki sayı girin (ör: '4,6') veya çıkmak için 'q': ")
    if user_input.lower() == 'q':
        print("Program sonlandırılıyor...")
        # Modeli kaydedip kaydetmemeyi kullanıcıya sor
        save_choice = input("Model kaydedilsin mi? (y/n): ").strip().lower()
        if save_choice == 'y':
            # Modeli ağırlıkları kaydetme
            np.savez('model.npz', W1=W1, b1=b1, W2=W2, b2=b2)
            print("Model ağırlıkları kaydedildi.")
        else:
            print("Model kaydedilmedi.")
        break
    try:
        a, b = map(int, user_input.split(','))
        user_pred = predict(a,b)
        print(f"{a} * {b} = {a*b}, Model Tahmini: {user_pred:.2f}")
    except:
        print("Geçersiz giriş formatı. Lütfen virgül ile ayrılmış iki tam sayı giriniz.")
