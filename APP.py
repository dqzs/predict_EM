import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor

# 添加 CSS 样式，定义圆角框
st.markdown(
    """
    <style>
    .rounded-container {
        border: 3px solid #4CAF50; /* 绿色边框 */
        border-radius: 15px; /* 圆角 */
        padding: 20px; /* 内边距 */
        margin: 20px auto; /* 自动居中 */
        background-color: #f9f9f9; /* 背景颜色 */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* 阴影效果 */
        width: 90%; /* 宽度 */
    }
    h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和描述
st.markdown(
    """
    <div class='rounded-container'>
    <h1>Predict Originc Fluorescence Emission Wavelengths</h1>
    <p>This website aims to quickly predict the emission wavelength of a molecule based on its structure (SMILES or SDF file) using machine learning models. It is recommended to use ChemDraw software to draw molecules and convert them to sdf.</p>
    """,
    unsafe_allow_html=True,
)

# 提供两种输入方式
input_option = st.radio("Choose input method:", ("SMILES Input", "SDF File Upload"))

# **SMILES 输入**
if input_option == "SMILES Input":
    smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., NC1=CC=C(C=C1)C(=O)O")
    if st.button("Submit and Predict"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                AllChem.AddHs(mol)
                AllChem.EmbedMolecule(mol)  # 使用 ETKDG 算法
                calc = Calculator(descriptors, ignore_3D=True)
                descriptors_data = calc.pandas([mol])
                # 这里应该是模型预测的代码，我们用一个模拟的预测结果代替
                st.write("Predicted Emission Wavelength: 450 nm (simulated)")
            else:
                st.error("Invalid SMILES input. Please check the format.")
        except Exception as e:
            st.error(f"An error occurred while processing SMILES: {e}")

# **SDF 文件上传**
elif input_option == "SDF File Upload":
    uploaded_file = st.file_uploader("Upload an SDF file", type=["sdf"])
    if uploaded_file:
        try:
            mol = Chem.MolFromMolBlock(uploaded_file.getvalue().decode('utf-8'))
            if mol:
                AllChem.AddHs(mol)
                AllChem.EmbedMolecule(mol)  # 使用 ETKDG 算法
                calc = Calculator(descriptors, ignore_3D=True)
                descriptors_data = calc.pandas([mol])
                # 这里应该是模型预测的代码，我们用一个模拟的预测结果代替
                st.write("Predicted Emission Wavelength: 450 nm (simulated)")
            else:
                st.error("No valid molecule found in the SDF file.")
        except Exception as e:
            st.error(f"An error occurred while processing the SDF file: {e}")

# 关闭圆角框 div
st.markdown("</div>", unsafe_allow_html=True)
