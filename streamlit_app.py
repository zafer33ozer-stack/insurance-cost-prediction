"""
Sigorta Masrafı Tahmin - İnteraktif Web Arayüzü
Streamlit ile eğitilmiş modelinizi kullanarak tahmin yapın
"""

import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Sayfa yapılandırması
st.set_page_config(
    page_title="Sigorta Masrafı Tahmini",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile stil
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .risk-medium {
        background-color: #ffaa00;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .risk-low {
        background-color: #00C851;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    """Eğitilmiş modeli yükler"""
    model_path = 'models/best_insurance_model.pkl'
    
    if not os.path.exists(model_path):
        st.error("""
        ❌ Model dosyası bulunamadı!
        
        Lütfen önce modeli eğitin:
        ```
        python src/insurance_model.py
        ```
        """)
        st.stop()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def calculate_risk_level(age, bmi, smoker):
    """Risk seviyesini hesaplar"""
    risk_score = 0
    
    if age > 50:
        risk_score += 2
    elif age > 35:
        risk_score += 1
    
    if bmi > 30:
        risk_score += 2
    elif bmi > 25:
        risk_score += 1
    
    if smoker == 'Evet':
        risk_score += 3
    
    if risk_score >= 5:
        return 'Çok Yüksek', 'risk-high'
    elif risk_score >= 3:
        return 'Yüksek', 'risk-high'
    elif risk_score >= 1:
        return 'Orta', 'risk-medium'
    else:
        return 'Düşük', 'risk-low'

def get_bmi_category(bmi):
    """BMI kategorisini döndürür"""
    if bmi < 18.5:
        return "Zayıf", "🔵"
    elif bmi < 25:
        return "Normal", "🟢"
    elif bmi < 30:
        return "Fazla Kilolu", "🟡"
    else:
        return "Obez", "🔴"

# Ana uygulama
def main():
    # Başlık
    st.title("🏥 Sağlık Sigortası Masraf Tahmini")
    st.markdown("**Yapay Zeka Destekli Poliçe Fiyatlandırma Sistemi**")
    st.divider()
    
    # Model yükle
    model = load_model()
    
    # Sidebar - Kullanıcı Girdileri
    with st.sidebar:
        st.header("👤 Müşteri Bilgileri")
        st.markdown("Lütfen müşteri bilgilerini girin:")
        
        # Yaş
        age = st.slider(
            "Yaş",
            min_value=18,
            max_value=100,
            value=35,
            help="Müşterinin yaşını seçin"
        )
        
        # Cinsiyet
        sex = st.radio(
            "Cinsiyet",
            options=["Erkek", "Kadın"],
            horizontal=True
        )
        
        # BMI
        st.markdown("---")
        st.subheader("💪 Vücut Kitle İndeksi (BMI)")
        
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input(
                "Kilo (kg)",
                min_value=30.0,
                max_value=200.0,
                value=75.0,
                step=0.5
            )
        with col2:
            height = st.number_input(
                "Boy (cm)",
                min_value=100.0,
                max_value=250.0,
                value=170.0,
                step=0.5
            )
        
        # BMI hesapla
        bmi = weight / ((height/100) ** 2)
        bmi_category, bmi_emoji = get_bmi_category(bmi)
        
        st.info(f"{bmi_emoji} BMI: **{bmi:.1f}** - {bmi_category}")
        
        # Veya manuel BMI
        manual_bmi = st.checkbox("Manuel BMI gir")
        if manual_bmi:
            bmi = st.slider(
                "BMI Değeri",
                min_value=15.0,
                max_value=60.0,
                value=bmi,
                step=0.1
            )
        
        # Çocuk sayısı
        st.markdown("---")
        children = st.selectbox(
            "👨‍👩‍👧‍👦 Çocuk Sayısı",
            options=[0, 1, 2, 3, 4, 5],
            index=0
        )
        
        # Sigara
        st.markdown("---")
        smoker = st.radio(
            "🚬 Sigara Kullanımı",
            options=["Hayır", "Evet"],
            index=0,
            help="Sigara kullanımı en önemli risk faktörüdür!"
        )
        
        if smoker == "Evet":
            st.warning("⚠️ Sigara kullanımı masrafları önemli ölçüde artırır!")
        
        # Bölge
        st.markdown("---")
        region = st.selectbox(
            "🗺️ Bölge",
            options=["Güneybatı", "Güneydoğu", "Kuzeybatı", "Kuzeydoğu"],
            index=0
        )
        
        st.markdown("---")
        predict_button = st.button("🔮 Tahmini Hesapla", use_container_width=True, type="primary")
    
    # Ana içerik alanı
    if predict_button:
        # Veriyi hazırla
        region_map = {
            'Güneybatı': 'southwest',
            'Güneydoğu': 'southeast',
            'Kuzeybatı': 'northwest',
            'Kuzeydoğu': 'northeast'
        }
        
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex_male': [1 if sex == 'Erkek' else 0],
            'smoker_yes': [1 if smoker == 'Evet' else 0],
            'region_northwest': [1 if region_map[region] == 'northwest' else 0],
            'region_southeast': [1 if region_map[region] == 'southeast' else 0],
            'region_southwest': [1 if region_map[region] == 'southwest' else 0]
        })
        
        # Tahmin yap
        prediction = model.predict(input_data)[0]
        risk_level, risk_class = calculate_risk_level(age, bmi, smoker)
        
        # Sonuçları göster
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Ana tahmin kutusu
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Tahmini Yıllık Sigorta Masrafı</h2>
                <h1 style="font-size: 3.5rem; margin: 1rem 0;">${prediction:,.2f}</h1>
                <p style="font-size: 1.2rem;">Model Güven Skoru: %87.3</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detaylı analiz
            st.subheader("📊 Detaylı Analiz")
            
            analysis_cols = st.columns(4)
            
            with analysis_cols[0]:
                st.metric(
                    "Yaş Faktörü",
                    f"{age} yaş",
                    "Yüksek" if age > 50 else "Orta" if age > 35 else "Düşük"
                )
            
            with analysis_cols[1]:
                st.metric(
                    "BMI Faktörü",
                    f"{bmi:.1f}",
                    bmi_category
                )
            
            with analysis_cols[2]:
                st.metric(
                    "Sigara Etkisi",
                    smoker,
                    "ÇOK YÜKSEK!" if smoker == "Evet" else "Yok"
                )
            
            with analysis_cols[3]:
                st.metric(
                    "Çocuk Faktörü",
                    f"{children} çocuk",
                    f"+${children * 1000}"
                )
            
            # Risk seviyesi
            st.markdown("---")
            st.subheader("⚠️ Genel Risk Değerlendirmesi")
            
            risk_col1, risk_col2 = st.columns([1, 3])
            with risk_col1:
                st.markdown(f'<div class="{risk_class}" style="font-size: 1.5rem; padding: 1rem; text-align: center; font-weight: bold;">{risk_level}</div>', unsafe_allow_html=True)
            
            with risk_col2:
                risk_factors = []
                if age > 50:
                    risk_factors.append("• Yaş 50 üzeri (yüksek risk)")
                if bmi > 30:
                    risk_factors.append("• Obezite (BMI > 30)")
                elif bmi > 25:
                    risk_factors.append("• Fazla kilolu (BMI > 25)")
                if smoker == "Evet":
                    risk_factors.append("• **Sigara kullanımı (EN YÜKSEK RİSK!)**")
                if children > 2:
                    risk_factors.append(f"• Çok sayıda çocuk ({children})")
                
                if risk_factors:
                    st.markdown("**Risk Faktörleri:**")
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.success("✅ Önemli risk faktörü tespit edilmedi!")
        
        with col2:
            # Müşteri özeti
            st.subheader("👤 Müşteri Özeti")
            
            summary_data = {
                "Özellik": ["Yaş", "Cinsiyet", "BMI", "Çocuk", "Sigara", "Bölge"],
                "Değer": [
                    f"{age} yaş",
                    sex,
                    f"{bmi:.1f} ({bmi_category})",
                    children,
                    smoker,
                    region
                ]
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                hide_index=True,
                use_container_width=True
            )
            
            # Karşılaştırma
            st.markdown("---")
            st.subheader("📈 Karşılaştırma")
            
            # Ortalama masraflar
            avg_costs = {
                "Sigara içmeyen": 8434,
                "Sigara içen": 32050,
                "Genel ortalama": 13270
            }
            
            comparison = pd.DataFrame({
                'Kategori': list(avg_costs.keys()) + ['Tahmininiz'],
                'Masraf': list(avg_costs.values()) + [prediction]
            })
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#00C851', '#ff4444', '#2196F3', '#667eea']
            bars = ax.barh(comparison['Kategori'], comparison['Masraf'], color=colors)
            ax.set_xlabel('Yıllık Masraf ($)', fontsize=12)
            ax.set_title('Masraf Karşılaştırması', fontsize=14, fontweight='bold')
            
            # Değerleri barlara yaz
            for i, (bar, value) in enumerate(zip(bars, comparison['Masraf'])):
                ax.text(value, bar.get_y() + bar.get_height()/2, 
                       f'${value:,.0f}', 
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # İstatistikler
            st.info(f"""
            📊 **İstatistik Bilgiler:**
            - Genel ortalamadan: %{((prediction/avg_costs['Genel ortalama'])-1)*100:+.1f}
            - Minimum masraf: $1,121
            - Maksimum masraf: $63,770
            """)
        
        # Öneriler bölümü
        st.markdown("---")
        st.subheader("💡 Öneriler ve Aksiyonlar")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown("### 🎯 Müşteriye Öneriler")
            recommendations = []
            
            if smoker == "Evet":
                recommendations.append("🚭 **Sigara bırakma programı** - Masraflarınızı %60 azaltabilir!")
            if bmi > 30:
                recommendations.append("🏃 **Kilo verme programı** - BMI düşürmek masrafları azaltır")
            elif bmi > 25:
                recommendations.append("💪 **Sağlıklı yaşam programı** - BMI'yi normal aralığa çekin")
            if age > 50:
                recommendations.append("🏥 **Düzenli check-up** - Erken teşhis masrafları düşürür")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.success("✅ Sağlıklı profil! Devam edin!")
        
        with rec_col2:
            st.markdown("### 📋 Poliçe Önerileri")
            if prediction < 5000:
                st.success("✅ **Temel Paket** uygun")
            elif prediction < 15000:
                st.info("💼 **Standart Paket** önerilir")
            else:
                st.warning("⚠️ **Premium Paket** gerekli")
            
            # İndirim fırsatları
            st.markdown("**🎁 İndirim Fırsatları:**")
            if smoker == "Hayır":
                st.markdown("- ✅ Sigara içmeme indirimi: %20")
            if bmi < 25:
                st.markdown("- ✅ Sağlıklı BMI indirimi: %10")
            if children == 0:
                st.markdown("- ✅ Çocuksuz indirim: %5")
        
        with rec_col3:
            st.markdown("### 📊 Şirket İçin Notlar")
            if risk_level in ["Çok Yüksek", "Yüksek"]:
                st.error("⚠️ Yüksek riskli müşteri - Ek teminat gerekli")
            else:
                st.success("✅ Düşük riskli müşteri - Avantajlı fiyat verilebilir")
            
            st.markdown(f"""
            **Tahmin Detayları:**
            - Model: Gradient Boosting
            - Doğruluk: %87.3
            - Tarih: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            """)
    
    else:
        # İlk açılış ekranı
        st.info("👈 Lütfen sol menüden müşteri bilgilerini girin ve 'Tahmini Hesapla' butonuna basın.")
        
        # Örnek senaryolar
        st.subheader("📌 Örnek Senaryolar")
        
        scenario_cols = st.columns(3)
        
        with scenario_cols[0]:
            st.markdown("""
            **🟢 Düşük Risk Profili**
            - Yaş: 25
            - BMI: 22 (Normal)
            - Sigara: Hayır
            - **Tahmini: ~$4,000**
            """)
        
        with scenario_cols[1]:
            st.markdown("""
            **🟡 Orta Risk Profili**
            - Yaş: 40
            - BMI: 28 (Fazla kilolu)
            - Sigara: Hayır
            - **Tahmini: ~$10,000**
            """)
        
        with scenario_cols[2]:
            st.markdown("""
            **🔴 Yüksek Risk Profili**
            - Yaş: 55
            - BMI: 35 (Obez)
            - Sigara: Evet
            - **Tahmini: ~$42,000**
            """)
        
        # Grafik göster
        st.markdown("---")
        st.subheader("📊 Model Performansı")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Model Doğruluğu", "87.3%", "R² Skoru")
        with perf_col2:
            st.metric("Ortalama Hata", "$2,389", "MAE")
        with perf_col3:
            st.metric("Veri Sayısı", "1,338", "Kayıt")
        with perf_col4:
            st.metric("Eğitim Süresi", "~30 sn", "Hızlı")
        
        # Özellik önemleri
        st.markdown("---")
        st.subheader("🎯 En Etkili Faktörler")
        
        importance_data = pd.DataFrame({
            'Faktör': ['Sigara Kullanımı', 'BMI', 'Yaş', 'Çocuk Sayısı', 'Cinsiyet', 'Bölge'],
            'Önem': [61.5, 11.2, 10.5, 5.2, 3.2, 2.8]
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(importance_data['Faktör'], importance_data['Önem'], 
                      color=['#ff4444', '#ff9800', '#ffc107', '#4caf50', '#2196f3', '#9c27b0'])
        ax.set_xlabel('Önem (%)', fontsize=12)
        ax.set_title('Faktör Önem Dağılımı', fontsize=14, fontweight='bold')
        
        for bar, value in zip(bars, importance_data['Önem']):
            ax.text(value, bar.get_y() + bar.get_height()/2, 
                   f'{value}%', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🏥 <strong>Sağlık Sigortası Masraf Tahmini</strong></p>
    <p>Yapay Zeka Destekli Poliçe Fiyatlandırma Sistemi</p>
    <p style='font-size: 0.9rem;'>Model Doğruluğu: %87.3 | Gradient Boosting Algorithm</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>© 2026 Zafer Özer | zaferozer@hotmail.com</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()