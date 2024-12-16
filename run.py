import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

# Tek dosyada kaydedilen modeli yükleme
model = np.load('model.npz')
W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']

def predict(x1, x2):
    # Girdileri normalizasyon ile aynı biçime getir
    inp = np.array([[x1/9.0, x2/9.0]])
    # İleri besleme (Forward Pass)
    Z1 = np.dot(inp, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    # Çıktıyı tekrar orijinal ölçeğine döndür (0-81 ölçeğinden geri çevir)
    return Z2[0,0]*81

# Kullanıcı etkileşimi
print("Kaydedilmiş modeli kullanarak tahmin yapma. (Çıkmak için 'q' giriniz.)")
while True:
    user_input = input("İki sayı girin (ör: '4,6') veya 'q' ile çıkın: ")
    if user_input.lower() == 'q':
        print("Program sonlandırılıyor...")
        break
    try:
        a, b = map(int, user_input.split(','))
        user_pred = predict(a,b)
        print(f"{a} * {b} = {a*b}, Model Tahmini: {user_pred:.2f}")
    except ValueError:
        print("Geçersiz giriş formatı. Lütfen virgül ile ayrılmış iki tam sayı giriniz.")
