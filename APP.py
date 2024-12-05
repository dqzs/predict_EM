import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile

# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto; /* 调整顶部外边距 */
        max-width: 39%; /* 使用百分比而不是固定值 */
        background-color: #f9f9f9;
    }
    @media (max-width: 768px) {
        .stApp {
            margin-top: 45px; /* 在小屏幕上减少顶部外边距 */
            max-width: 95%; /* 在小屏幕上使用全宽 */
        }
    }
    .rounded-container h2 {
        margin-top: -80px; /* 调整顶部外边距 */
        text-align: center; /* 确保标题文本居中 */
        background-color: #e0e0e0; /* 标题背景颜色 */
        padding: 10px; /* 标题内边距 */
        border-radius: 10px; /* 标题圆角 */
    }
    .rounded-container blockquote {
        text-align: left; /* 保持块引用文本左对齐 */
        margin: 20px auto; /* 调整顶部外边距 */
        background-color: #f0f0f0; /* 块引用背景颜色 */
        padding: 10px; /* 块引用内边距 */
        font-size: 1.1em; /* 字体大小 */
        border-radius: 10px; /* 标题圆角 */
    }
    a {
        color: #0000EE; /* 链接颜色 */
        text-decoration: underline; /* 链接下划线 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Organic Fluorescence Emission Wavelengths</h2>
        <blockquote>
            1. This website aims to quickly predict the emission wavelength of organic molecules based on their structure (SMILES or SDF files) using machine learning models.<br>
            2. It is recommended to use ChemDraw software to draw the molecular structure and convert it to sdf.<br>
            3. Code and data are available at <a href='https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction' target='_blank'>GitHub</a>. 
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 提供两种输入方式
input_option = st.radio("Choose input method:", ("SMILES Input", "SDF File Upload"))
mols = []  # 存储分子列表

# SMILES 输入区域
if input_option == "SMILES Input":
    smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., NC1=CC=C(C=C1)C(=O)O")

# SDF 文件上传区域
elif input_option == "SDF File Upload":
    uploaded_file = st.file_uploader("Upload an SDF file", type=["sdf"])

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")
# 用户指定的描述符列表
required_descriptors = [
    "SdsCH", "MolLogP", "SdssC", "VSA_EState7",
    "SlogP_VSA8", "VE1_A", "EState_VSA4", "AATS8i", "AATS4i"
]

# 如果点击提交按钮
if submit_button:
    # 根据用户选择的输入方式处理数据
    if input_option == "SMILES Input" and smiles:
        try:
            st.text("Processing SMILES input...")  # 显示处理信息
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                AllChem.AddHs(mol)
                AllChem.EmbedMolecule(mol)  # 使用 ETKDG 算法
                mols.append(mol)
            else:
                st.error("Invalid SMILES input. Please check the format.")
        except Exception as e:
            st.error(f"An error occurred while processing SMILES: {e}")

    elif input_option == "SDF File Upload" and uploaded_file:
        try:
            st.text("Processing SDF file...")  # 显示处理信息
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filename = temp_file.name

            supplier = Chem.SDMolSupplier(temp_filename)
            for mol in supplier:
                if mol is not None:
                    mols.append(mol)
                    break  # 仅加载第一个分子

            if len(mols) > 0:
                st.success("File uploaded successfully, containing 1 valid molecule!")
            else:
                st.error("No valid molecule found in the SDF file!")
        except Exception as e:
            st.error(f"An error occurred while processing the SDF file: {e}")

# 如果点击提交按钮且存在有效分子
if submit_button and mols:
    with st.spinner("Calculating molecular descriptors and making predictions..."):
        try:
            st.info("Calculating molecular descriptors, please wait...")
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_selected = [d for d in calc.descriptors if str(d) in required_descriptors]
            if mordred_selected:
                calc = Calculator(mordred_selected, ignore_3D=True)
            rdkit_descriptor_map = {
                "MolLogP": Descriptors.MolLogP,
                "MolWt": Descriptors.MolWt,
            }
            rdkit_selected = {k: v for k, v in rdkit_descriptor_map.items() if k in required_descriptors or k == "MolWt"}
            molecular_descriptor = []
            for mol in mols:
                if mol is None:
                    continue
                rdkit_desc_dict = {name: func(mol) for name, func in rdkit_selected.items()}
                rdkit_desc_df = pd.DataFrame([rdkit_desc_dict])
                mol_weight = rdkit_desc_dict.get("MolWt", "N/A")
                st.write(f"Molecular Weight: {mol_weight:.2f} g/mol")
                mordred_desc_df = calc.pandas([mol])
                combined_descriptors = mordred_desc_df.join(rdkit_desc_df)
                molecular_descriptor.append(combined_descriptors)
            result_df = pd.concat(molecular_descriptor, ignore_index=True)
            st.info("Loading the model and predicting the emission wavelength, please wait...")
            predictor = TabularPredictor.load("ag-20241119_124834")
            model_options = [
                "LightGBM_BAG_L1", "LightGBMXT_BAG_L1", "CatBoost_BAG_L1",
                "NeuralNetTorch_BAG_L1", "LightGBMLarge_BAG_L1", "WeightedEnsemble_L2"
            ]
            predictions_dict = {}
            for model in model_options:
                predictions = predictor.predict(result_df, model=model)
                predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")
            st.write("Prediction results from all models:")
            results_df = pd.DataFrame(predictions_dict)
            results_df["Molecule Index"] = range(len(mols))
            results_df = results_df[["Molecule Index"] + model_options]
            st.dataframe(results_df)
        except Exception as e:
            st.error(f"An error occurred during molecular descriptor calculation or prediction: {e}")
