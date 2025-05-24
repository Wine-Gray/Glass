import streamlit as st
import joblib
import numpy as np

# ===== 加载三个模型 =====
model_density = joblib.load('rf_density_model.pkl')
model_strength = joblib.load('rf_strength_model.pkl')
model_thermal = joblib.load('rf_thermal_model.pkl')

# ===== Streamlit界面设计 =====
st.title("陶瓷性能预测系统")
st.write("输入陶瓷主要成分（占比 %），一键获得致密度、抗弯强度和热导率预测结果")

# ===== 用户输入区 =====
al2o3 = st.number_input("Al₂O₃ (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
sio2  = st.number_input("SiO₂ (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
fe2o3 = st.number_input("Fe₂O₃ (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.01)
mgo   = st.number_input("MgO (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.01)

# ===== 预测按钮 =====
if st.button("一键预测三大性能指标"):
    input_data = np.array([[al2o3, sio2, fe2o3, mgo]])
    pred_density = model_density.predict(input_data)[0]
    pred_strength = model_strength.predict(input_data)[0]
    pred_thermal = model_thermal.predict(input_data)[0]
    
    st.success(f"预测致密度：{pred_density:.3f} g/cm³")
    st.success(f"预测抗弯强度：{pred_strength:.2f} MPa")
    st.success(f"预测热导率：{pred_thermal:.3f} W/m·K")
