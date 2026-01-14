"""
Sigorta Masrafı Tahmin Modülü
Eğitilmiş model ile yeni müşteriler için tahmin yapar
"""

import pickle
import pandas as pd
import os
from typing import Union

class InsurancePredictor:
    """Sigorta masrafı tahmin sınıfı"""
    
    def __init__(self, model_path: str = 'models/best_insurance_model.pkl'):
        """
        Args:
            model_path: Eğitilmiş modelin dosya yolu
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Eğitilmiş modeli yükler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {self.model_path}\n"
                "Lütfen önce 'python src/insurance_model.py' çalıştırın."
            )
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Özellik isimlerini yükle
        encoder_path = 'models/encoders.pkl'
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                encoders = pickle.load(f)
                self.feature_names = encoders.get('feature_names', None)
        
        print(f"✓ Model başarıyla yüklendi: {self.model_path}")
    
    def prepare_input(self, age: int, sex: str, bmi: float, 
                     children: int, smoker: str, region: str) -> pd.DataFrame:
        """
        Girdi verilerini model için hazırlar
        
        Args:
            age: Yaş (18-100)
            sex: Cinsiyet ('male' veya 'female')
            bmi: Vücut Kitle İndeksi (15-60)
            children: Çocuk sayısı (0-10)
            smoker: Sigara kullanımı ('yes' veya 'no')
            region: Bölge ('northeast', 'northwest', 'southeast', 'southwest')
        
        Returns:
            Model için hazırlanmış DataFrame
        """
        # Validasyon
        if not 18 <= age <= 100:
            raise ValueError("Yaş 18-100 arasında olmalıdır")
        if sex not in ['male', 'female']:
            raise ValueError("Cinsiyet 'male' veya 'female' olmalıdır")
        if not 15 <= bmi <= 60:
            raise ValueError("BMI 15-60 arasında olmalıdır")
        if not 0 <= children <= 10:
            raise ValueError("Çocuk sayısı 0-10 arasında olmalıdır")
        if smoker not in ['yes', 'no']:
            raise ValueError("Sigara kullanımı 'yes' veya 'no' olmalıdır")
        if region not in ['northeast', 'northwest', 'southeast', 'southwest']:
            raise ValueError("Geçersiz bölge")
        
        # One-hot encoding ile veri hazırlama
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex_male': [1 if sex == 'male' else 0],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        })
        
        return input_data
    
    def predict(self, age: int, sex: str, bmi: float, 
                children: int, smoker: str, region: str) -> float:
        """
        Sigorta masrafını tahmin eder
        
        Returns:
            Tahmini yıllık sigorta masrafı (USD)
        """
        input_data = self.prepare_input(age, sex, bmi, children, smoker, region)
        prediction = self.model.predict(input_data)[0]
        return round(prediction, 2)
    
    def predict_with_details(self, age: int, sex: str, bmi: float, 
                           children: int, smoker: str, region: str) -> dict:
        """
        Detaylı tahmin bilgisi döndürür
        
        Returns:
            Dict ile tahmin ve risk faktörleri
        """
        prediction = self.predict(age, sex, bmi, children, smoker, region)
        
        # Risk faktör analizi
        risk_factors = {
            'Yaş Riski': 'Yüksek' if age > 50 else 'Orta' if age > 35 else 'Düşük',
            'BMI Riski': 'Yüksek' if bmi > 30 else 'Orta' if bmi > 25 else 'Düşük',
            'Sigara Riski': 'Çok Yüksek' if smoker == 'yes' else 'Yok',
            'Genel Risk Seviyesi': self._calculate_risk_level(age, bmi, smoker)
        }
        
        return {
            'tahmini_masraf': prediction,
            'risk_faktörleri': risk_factors,
            'girdi_bilgileri': {
                'Yaş': age,
                'Cinsiyet': 'Erkek' if sex == 'male' else 'Kadın',
                'BMI': bmi,
                'Çocuk Sayısı': children,
                'Sigara': 'Evet' if smoker == 'yes' else 'Hayır',
                'Bölge': region
            }
        }
    
    def _calculate_risk_level(self, age: int, bmi: float, smoker: str) -> str:
        """Genel risk seviyesini hesaplar"""
        risk_score = 0
        
        if age > 50:
            risk_score += 2
        elif age > 35:
            risk_score += 1
        
        if bmi > 30:
            risk_score += 2
        elif bmi > 25:
            risk_score += 1
        
        if smoker == 'yes':
            risk_score += 3
        
        if risk_score >= 5:
            return 'Çok Yüksek'
        elif risk_score >= 3:
            return 'Yüksek'
        elif risk_score >= 1:
            return 'Orta'
        else:
            return 'Düşük'


def predict_insurance_cost(age: int, sex: str, bmi: float, 
                          children: int, smoker: str, 
                          region: str = 'southwest') -> float:
    """
    Hızlı tahmin fonksiyonu (convenience function)
    
    Args:
        age: Yaş
        sex: Cinsiyet ('male' veya 'female')
        bmi: Vücut Kitle İndeksi
        children: Çocuk sayısı
        smoker: Sigara kullanımı ('yes' veya 'no')
        region: Bölge (varsayılan 'southwest')
    
    Returns:
        Tahmini yıllık sigorta masrafı (USD)
    
    Example:
        >>> cost = predict_insurance_cost(35, 'male', 27.5, 2, 'no')
        >>> print(f"Tahmin: ${cost:,.2f}")
    """
    predictor = InsurancePredictor()
    return predictor.predict(age, sex, bmi, children, smoker, region)


if __name__ == '__main__':
    # Test örnekleri
    print("=" * 60)
    print("SİGORTA MASRAFI TAHMİN UYGULAMASI")
    print("=" * 60)
    
    predictor = InsurancePredictor()
    
    # Örnek 1: Düşük risk
    print("\n📋 Örnek 1: Düşük Risk Profili")
    print("-" * 60)
    result1 = predictor.predict_with_details(
        age=25, sex='female', bmi=22, children=0, 
        smoker='no', region='southwest'
    )
    print(f"Tahmini Masraf: ${result1['tahmini_masraf']:,.2f}")
    print(f"Risk Seviyesi: {result1['risk_faktörleri']['Genel Risk Seviyesi']}")
    
    # Örnek 2: Orta risk
    print("\n📋 Örnek 2: Orta Risk Profili")
    print("-" * 60)
    result2 = predictor.predict_with_details(
        age=40, sex='male', bmi=28, children=2, 
        smoker='no', region='northeast'
    )
    print(f"Tahmini Masraf: ${result2['tahmini_masraf']:,.2f}")
    print(f"Risk Seviyesi: {result2['risk_faktörleri']['Genel Risk Seviyesi']}")
    
    # Örnek 3: Yüksek risk
    print("\n📋 Örnek 3: Yüksek Risk Profili")
    print("-" * 60)
    result3 = predictor.predict_with_details(
        age=55, sex='male', bmi=35, children=3, 
        smoker='yes', region='southeast'
    )
    print(f"Tahmini Masraf: ${result3['tahmini_masraf']:,.2f}")
    print(f"Risk Seviyesi: {result3['risk_faktörleri']['Genel Risk Seviyesi']}")
    print(f"\n⚠️ Sigara Riski: {result3['risk_faktörleri']['Sigara Riski']}")
    
    print("\n" + "=" * 60)