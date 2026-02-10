"""
Improved Insurance Cost Prediction - Streamlit App (Enhanced UI)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Custom CSS for better styling
# ------------------------------------------------------------------
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Metric cards styling */
        [data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #667eea;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f7fa 0%, #e9ecef 100%);
        }
        
        /* Title styling */
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1.5rem;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
            margin-top: 1.5rem;
        }
        
        /* Input labels styling */
        label {
            font-weight: 600;
            color: #2c3e50;
            margin-top: 0.8rem;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 700;
            font-size: 1.1rem;
            padding: 0.75rem;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* Column styling */
        .stColumn {
            padding: 1rem;
        }
        
        /* Divider */
        hr {
            border: 0;
            height: 2px;
            background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
            margin: 2rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load pre-trained model and metadata"""
    try:
        model = joblib.load("models/best_model_with_fe.joblib")
        info = joblib.load("models/model_info_fe.joblib")
        comparison = joblib.load("models/comparison_results.joblib")
        return model, info, comparison
    except FileNotFoundError:
        return None, None, None

# ------------------------------------------------------------------
# Feature engineering (must match training exactly!)
# ------------------------------------------------------------------
def build_features(age, bmi, children, sex, smoker, region):
    """Build feature dictionary for model prediction"""
    sex_male = 1 if sex == "Male" else 0
    smoker_yes = 1 if smoker == "Yes" else 0
    region_map = {
        "Southwest": (0, 0, 1),
        "Southeast": (0, 1, 0),
        "Northwest": (1, 0, 0),
        "Northeast": (0, 0, 0)
    }
    region_nw, region_se, region_sw = region_map[region]
    
    return {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": sex_male,
        "smoker_yes": smoker_yes,
        "region_northwest": region_nw,
        "region_southeast": region_se,
        "region_southwest": region_sw,
        "bmi_smoker": bmi * smoker_yes,
        "is_obese": int(bmi > 30),
        "age_group": 0 if age <= 25 else 1 if age <= 35 else 2 if age <= 50 else 3,
        "bmi_category": 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3,
        "high_risk": int(smoker_yes == 1 and bmi > 30),
        "age_bmi": age * bmi,
        "has_children": int(children > 0)
    }

def get_health_category(bmi):
    """Determine health category based on BMI"""
    if bmi < 18.5:
        return "🟦 Underweight"
    elif bmi < 25:
        return "🟩 Normal Weight"
    elif bmi < 30:
        return "🟨 Overweight"
    else:
        return "🟥 Obese"

def get_risk_level(age, bmi, smoker):
    """Determine risk level based on factors"""
    risk_score = 0
    
    if age > 50:
        risk_score += 2
    elif age > 40:
        risk_score += 1
        
    if bmi > 30:
        risk_score += 2
    elif bmi > 25:
        risk_score += 1
        
    if smoker == "Yes":
        risk_score += 3
    
    if risk_score >= 5:
        return "🔴 High Risk", risk_score
    elif risk_score >= 3:
        return "🟠 Medium Risk", risk_score
    else:
        return "🟢 Low Risk", risk_score

# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    # Header
    st.markdown("<h1>🏥 Insurance Cost Prediction System</h1>", unsafe_allow_html=True)
    
    # Check if models exist
    model, info, comparison = load_artifacts()
    if model is None:
        st.error("❌ Model files not found. Please run the training script first.")
        st.stop()
    
    # Performance Metrics Section
    st.markdown("<h2>📊 Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Original R² Score",
            value=f"{comparison['original_r2']:.4f}",
            delta="Baseline"
        )
    
    with col2:
        st.metric(
            label="Improved R² Score",
            value=f"{comparison['best_r2']:.4f}",
            delta="Enhanced"
        )
    
    with col3:
        improvement_pct = comparison['improvement']
        st.metric(
            label="Performance Gain",
            value=f"{improvement_pct:.2f}%",
            delta="Improvement"
        )
    
    with col4:
        st.metric(
            label="Model Status",
            value="Active",
            delta="Ready"
        )
    
    st.markdown("---")
    
    # Main prediction section
    st.markdown("<h2>💻 Make a Prediction</h2>", unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📋 Patient Information")
        st.markdown("---")
        
        # Personal Information
        st.markdown("**👤 Personal Details**")
        age = st.slider("Age (years)", 18, 100, 35, help="Select your age")
        sex = st.radio("Sex", ["Male", "Female"], help="Select your biological sex")
        smoker = st.radio("Smoker", ["No", "Yes"], help="Do you smoke?")
        children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5], help="How many dependents?")
        
        st.markdown("---")
        
        # Location
        st.markdown("**📍 Location**")
        region = st.selectbox(
            "Region",
            ["Southwest", "Southeast", "Northwest", "Northeast"],
            help="Select your geographic region"
        )
        
        st.markdown("---")
        
        # Physical Measurements
        st.markdown("**⚖️ Physical Measurements**")
        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=75.0,
            step=0.5,
            help="Your body weight in kilograms"
        )
        height = st.number_input(
            "Height (cm)",
            min_value=120.0,
            max_value=220.0,
            value=170.0,
            step=0.5,
            help="Your height in centimeters"
        )
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        st.markdown("---")
        
        # Prediction button
        predict_btn = st.button("🔮 Predict Insurance Cost", use_container_width=True)
    
    # Main content area with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Your Health Profile")
        
        # Create health profile card
        profile_data = {
            "Age": f"{age} years",
            "BMI": f"{bmi:.2f}",
            "Health Category": get_health_category(bmi),
            "Smoking Status": "🚬 Smoker" if smoker == "Yes" else "🚭 Non-smoker",
            "Dependents": f"{children} child(ren)",
            "Region": region
        }
        
        for key, value in profile_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown("### ⚠️ Risk Assessment")
        
        risk_level, risk_score = get_risk_level(age, bmi, smoker)
        st.markdown(f"### {risk_level}")
        st.write(f"Risk Score: {risk_score}/10")
        
        # Risk factors breakdown
        st.markdown("**Risk Factors:**")
        risk_factors = []
        
        if age > 50:
            risk_factors.append("🔸 Age > 50 years")
        if bmi > 30:
            risk_factors.append("🔸 BMI indicates obesity")
        if smoker == "Yes":
            risk_factors.append("🔸 Smoking status")
        if children > 3:
            risk_factors.append("🔸 Multiple dependents")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("✅ No major risk factors detected")
    
    st.markdown("---")
    
    # Prediction result
    if predict_btn:
        with st.spinner("🔄 Calculating prediction..."):
            features = build_features(age, bmi, children, sex, smoker, region)
            df = pd.DataFrame([features], columns=info["feature_names"])
            prediction = model.predict(df)[0]
            
            # Display prediction result
            st.markdown("<h2>🎯 Prediction Result</h2>", unsafe_allow_html=True)
            
            # Large success box with prediction
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                    ">
                        <h3 style="margin: 0; font-size: 1.2rem; color: rgba(255,255,255,0.9);">
                            Estimated Annual Cost
                        </h3>
                        <h1 style="margin: 0.5rem 0; font-size: 3rem;">
                            ${prediction:,.2f}
                        </h1>
                        <p style="margin: 0; color: rgba(255,255,255,0.8);">
                            Per Year
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                monthly_cost = prediction / 12
                st.markdown(f"""
                    <div style="
                        background: rgba(102, 126, 234, 0.1);
                        padding: 2rem;
                        border-radius: 15px;
                        border-left: 5px solid #667eea;
                    ">
                        <h4>📅 Monthly Breakdown</h4>
                        <h2 style="color: #667eea; margin: 0.5rem 0;">
                            ${monthly_cost:,.2f}
                        </h2>
                        <p style="color: #666; margin: 0;">Per Month</p>
                        
                        <hr style="margin: 1rem 0; border: none; height: 1px; background: #ddd;">
                        
                        <h4>💡 Quick Facts</h4>
                        <ul style="color: #555; margin: 0;">
                            <li>Weekly: ${prediction/52:,.2f}</li>
                            <li>Daily: ${prediction/365:,.2f}</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div style="
                        background: rgba(102, 126, 234, 0.1);
                        padding: 1.5rem;
                        border-radius: 10px;
                        border-left: 5px solid #667eea;
                        text-align: center;
                    ">
                        <h4>📊 BMI Impact</h4>
                        <p style="font-size: 1.5rem; color: #667eea; margin: 0.5rem 0;">
                            {bmi:.1f}
                        </p>
                        <small>{get_health_category(bmi)}</small>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div style="
                        background: rgba(102, 126, 234, 0.1);
                        padding: 1.5rem;
                        border-radius: 10px;
                        border-left: 5px solid #667eea;
                        text-align: center;
                    ">
                        <h4>🎯 Risk Level</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                            {get_risk_level(age, bmi, smoker)[0]}
                        </p>
                        <small>Overall Assessment</small>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div style="
                        background: rgba(102, 126, 234, 0.1);
                        padding: 1.5rem;
                        border-radius: 10px;
                        border-left: 5px solid #667eea;
                        text-align: center;
                    ">
                        <h4>👤 Age Group</h4>
                        <p style="font-size: 1.5rem; color: #667eea; margin: 0.5rem 0;">
                            {age} yrs
                        </p>
                        <small>Years Old</small>
                    </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("<h3>💭 Health Recommendations</h3>", unsafe_allow_html=True)
            
            recommendations = []
            
            if bmi > 30:
                recommendations.append("🥗 **Weight Management**: Consider a balanced diet and regular exercise to reduce BMI.")
            if smoker == "Yes":
                recommendations.append("🚭 **Quit Smoking**: This is the most impactful change you can make for your health and insurance costs.")
            if age > 50:
                recommendations.append("🏥 **Regular Checkups**: Schedule regular health screenings appropriate for your age.")
            if children > 0:
                recommendations.append("👨‍👩‍👧 **Family Health**: Ensure all dependents have adequate health coverage.")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ **Great job!** Your health profile looks good. Continue maintaining healthy habits.")

if __name__ == "__main__":
    main()