import streamlit as st

# 设置页面配置
st.set_page_config(page_title="Predict Fluorescence Emission Wavelengths", layout="wide")

# 使用 markdown 和 HTML 创建自定义布局
html_temp = """
<div style='padding: 20px; border: 3px solid #4CAF50; border-radius: 15px; background-color: #f9f9f9; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
    <h1 style='text-align: center;'>Predict Originc Fluorescence Emission Wavelengths</h1>
    <p style='text-align: center;'>This website aims to quickly predict the emission wavelength of a molecule based on its structure (SMILES or SDF file) using machine learning models. It is recommended to use ChemDraw software to draw molecules and convert them to sdf.</p>
    <div style='margin: 20px;'>
        <form>
            <label for="input_method">Choose input method:</label><br>
            <input type="radio" id="smiles" name="input_method" value="smiles" checked>
            <label for="smiles">SMILES Input</label><br>
            <input type="radio" id="sdf" name="input_method" value="sdf">
            <label for="sdf">SDF File Upload</label><br><br>
            <label for="smiles_input">Enter the SMILES representation of the molecule:</label><br>
            <input type="text" id="smiles_input" name="smiles_input" placeholder="e.g., NC1=CC=C(C=C1)C(=O)O"><br><br>
            <button type="button" onclick="alert('Submit and Predict')">Submit and Predict</button>
        </form>
    </div>
</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)
