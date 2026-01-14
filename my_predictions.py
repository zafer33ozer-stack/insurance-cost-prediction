from src.predict import predict_insurance_cost

# Hızlı tahmin
cost = predict_insurance_cost(
    age=35,
    sex='male',
    bmi=27.5,
    children=2,
    smoker='no'
)
print(f"Tahmin: ${cost:,.2f}")

# Detaylı tahmin
from src.predict import InsurancePredictor

predictor = InsurancePredictor()
result = predictor.predict_with_details(
    age=45, sex='female', bmi=32, 
    children=1, smoker='yes', region='northeast'
)

print(f"\n💰 Masraf: ${result['tahmini_masraf']:,.2f}")
print(f"🎯 Risk: {result['risk_faktörleri']['Genel Risk Seviyesi']}")
print(f"⚠️  Sigara: {result['risk_faktörleri']['Sigara Riski']}")