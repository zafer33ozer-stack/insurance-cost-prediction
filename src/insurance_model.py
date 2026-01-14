# Sigorta Masrafı Tahmin Modeli
# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Veri setini yükleme
df = pd.read_csv('data/insurance.csv')

print("=" * 60)
print("SİGORTA MASRAFI TAHMİN PROJESİ")
print("=" * 60)

# 1. VERİ KEŞFİ VE ANALİZ
print("\n1. VERİ SETİ GENEL BAKIŞ")
print("-" * 60)
print(f"Veri seti boyutu: {df.shape}")
print(f"\nİlk 5 kayıt:\n{df.head()}")
print(f"\nVeri tipleri:\n{df.dtypes}")
print(f"\nEksik değerler:\n{df.isnull().sum()}")
print(f"\nTemel istatistikler:\n{df.describe()}")

# 2. VERİ GÖRSELLEŞTİRME
print("\n2. VERİ GÖRSELLEŞTİRME")
print("-" * 60)

# 2.1 Hedef değişken dağılımı
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(df['charges'], bins=50, edgecolor='black', color='skyblue')
plt.title('Sigorta Masrafı Dağılımı')
plt.xlabel('Masraf ($)')
plt.ylabel('Frekans')

# 2.2 Yaş vs Masraf
plt.subplot(2, 3, 2)
plt.scatter(df['age'], df['charges'], alpha=0.5)
plt.title('Yaş vs Masraf')
plt.xlabel('Yaş')
plt.ylabel('Masraf ($)')

# 2.3 BMI vs Masraf
plt.subplot(2, 3, 3)
plt.scatter(df['bmi'], df['charges'], alpha=0.5, color='green')
plt.title('BMI vs Masraf')
plt.xlabel('BMI')
plt.ylabel('Masraf ($)')

# 2.4 Sigara içenlerin etkisi
plt.subplot(2, 3, 4)
df.boxplot(column='charges', by='smoker', ax=plt.gca())
plt.title('Sigara Kullanımına Göre Masraf')
plt.suptitle('')
plt.xlabel('Sigara İçiyor mu?')
plt.ylabel('Masraf ($)')

# 2.5 Cinsiyet etkisi
plt.subplot(2, 3, 5)
df.boxplot(column='charges', by='sex', ax=plt.gca())
plt.title('Cinsiyete Göre Masraf')
plt.suptitle('')
plt.xlabel('Cinsiyet')
plt.ylabel('Masraf ($)')

# 2.6 Bölge etkisi
plt.subplot(2, 3, 6)
df.boxplot(column='charges', by='region', ax=plt.gca())
plt.title('Bölgeye Göre Masraf')
plt.suptitle('')
plt.xlabel('Bölge')
plt.ylabel('Masraf ($)')

plt.tight_layout()
plt.savefig('outputs/insurance_eda.png', dpi=300, bbox_inches='tight')
print("✓ Görselleştirmeler 'outputs/insurance_eda.png' dosyasına kaydedildi")

# 3. KORELASYON ANALİZİ
print("\n3. KORELASYON ANALİZİ")
print("-" * 60)

# Kategorik değişkenleri sayısala çevirme
df_encoded = df.copy()
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df_encoded['sex'] = le_sex.fit_transform(df_encoded['sex'])
df_encoded['smoker'] = le_smoker.fit_transform(df_encoded['smoker'])
df_encoded['region'] = le_region.fit_transform(df_encoded['region'])

correlation_matrix = df_encoded.corr()
print(f"\nKorelasyon Matrisi:\n{correlation_matrix['charges'].sort_values(ascending=False)}")

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Korelasyon Isı Haritası')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Korelasyon haritası 'outputs/correlation_heatmap.png' dosyasına kaydedildi")

# 4. VERİ HAZIRLIKLARI
print("\n4. VERİ HAZIRLAMA")
print("-" * 60)

# Kategorik değişkenleri encode etme (One-Hot Encoding)
df_model = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print(f"Model için hazırlanan veri seti boyutu: {df_model.shape}")
print(f"\nÖzellikler: {df_model.columns.tolist()}")

# Özellikler ve hedef değişken
X = df_model.drop('charges', axis=1)
y = df_model['charges']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# 5. MODEL EĞİTİMİ
print("\n5. MODEL EĞİTİMİ VE DEĞERLENDİRME")
print("-" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n{name} modeli eğitiliyor...")
    
    # Model eğitimi
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Performans metrikleri
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred_test
    }
    
    print(f"  Eğitim R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")

# En iyi modeli seçme
best_model_name = max(results, key=lambda x: results[x]['test_r2'])
print(f"\n{'='*60}")
print(f"EN İYİ MODEL: {best_model_name}")
print(f"Test R² Skoru: {results[best_model_name]['test_r2']:.4f}")
print(f"{'='*60}")

# 6. ÖZELLİK ÖNEMİ ANALİZİ
print("\n6. ÖZELLİK ÖNEMİ ANALİZİ")
print("-" * 60)

# Random Forest için özellik önemleri
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nÖzellik Önemleri (Random Forest):")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Önem Skoru')
plt.title('Özellik Önemleri')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Özellik önemleri 'outputs/feature_importance.png' dosyasına kaydedildi")

# 7. TAHMİN SONUÇLARININ GÖRSELLEŞTİRİLMESİ
print("\n7. TAHMİN SONUÇLARI")
print("-" * 60)

plt.figure(figsize=(15, 5))

for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(y_test, result['predictions'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Değerler ($)')
    plt.ylabel('Tahmin Edilen Değerler ($)')
    plt.title(f'{name}\nR² = {result["test_r2"]:.4f}')
    plt.tight_layout()

plt.savefig('outputs/predictions_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Tahmin karşılaştırmaları 'outputs/predictions_comparison.png' dosyasına kaydedildi")

# 8. ÖRNEK TAHMİN
print("\n8. ÖRNEK TAHMİN")
print("-" * 60)

# Yeni bir müşteri verisi
new_customer = pd.DataFrame({
    'age': [35],
    'bmi': [27.5],
    'children': [2],
    'sex_male': [1],
    'smoker_yes': [0],
    'region_northwest': [0],
    'region_southeast': [0],
    'region_southwest': [1]
})

best_model = results[best_model_name]['model']
predicted_cost = best_model.predict(new_customer)[0]

print("\nYeni Müşteri Bilgileri:")
print(f"  Yaş: 35")
print(f"  Cinsiyet: Erkek")
print(f"  BMI: 27.5")
print(f"  Çocuk Sayısı: 2")
print(f"  Sigara: Hayır")
print(f"  Bölge: Southwest")
print(f"\nTahmini Yıllık Masraf: ${predicted_cost:,.2f}")

# 9. MODEL KAYDETME
print("\n9. MODEL KAYDETME")
print("-" * 60)
import pickle

# En iyi modeli kaydetme
with open('models/best_insurance_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Model 'models/best_insurance_model.pkl' dosyasına kaydedildi")

# Encoderları kaydetme
encoders = {
    'feature_names': X.columns.tolist()
}
with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("✓ Encoder bilgileri 'models/encoders.pkl' dosyasına kaydedildi")

print("\n" + "="*60)
print("PROJE TAMAMLANDI!")
print("="*60)
print("\nOluşturulan Dosyalar:")
print("  1. outputs/insurance_eda.png - Veri analizi grafikleri")
print("  2. outputs/correlation_heatmap.png - Korelasyon haritası")
print("  3. outputs/feature_importance.png - Özellik önemleri")
print("  4. outputs/predictions_comparison.png - Model karşılaştırmaları")
print("  5. models/best_insurance_model.pkl - Eğitilmiş model")
print("  6. models/encoders.pkl - Encoder bilgileri")

plt.show()