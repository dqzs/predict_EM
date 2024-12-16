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
        margin: 50px auto;
        max-width: 39%; /* 设置最大宽度 */
        background-color: #f9f9f9f9;
        padding: 20px; /* 增加内边距 */
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.1em;
        border-radius: 10px;
    }
    a {
        color: #0000EE;
        text-decoration: underline;
    }
    .process-text, .molecular-weight {
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    .stDataFrame {
        margin-top: 10px;
        margin-bottom: 0px !important;
    }
     /* 针对小屏幕的优化 */
    @media (max-width: 768px) {
        .rounded-container {
            padding: 10px; /* 减少内边距 */
        }
        .rounded-container blockquote {
            font-size: 0.9em; /* 缩小字体 */
        }
        .rounded-container h2 {
            font-size: 1.2em; /* 调整标题字体大小 */
        }
        .stApp {
            margin: 0 !important; /* 移除外边距 */
            padding: 5px !important; /* 减少内边距 */
        }
        .process-text, .molecular-weight {
            font-size: 0.9em; /* 缩小文本字体 */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Organic Fluorescence <br>Emission Wavelengths</h2>
        <blockquote>
            1. This website aims to quickly predict the emission wavelength of organic molecules based on their structure (SMILES or SDF files) using machine learning models.<br>
            2. It is recommended to use ChemDraw software to draw the molecular structure and convert it to sdf.<br>
            3. Code and data are available at <a href='https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction' target='_blank'>https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction</a>.
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
    with st.container():  # 使用容器来限制宽度
        if input_option == "SMILES Input" and smiles:
            try:
                st.markdown('<div class="process-text">Processing SMILES input...</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="process-text">Processing SDF file...</div>', unsafe_allow_html=True)
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
                st.markdown(f'<div class="molecular-weight">Molecular Weight: {mol_weight:.2f} g/mol</div>', unsafe_allow_html=True)
                mordred_desc_df = calc.pandas([mol])
                combined_descriptors = mordred_desc_df.join(rdkit_desc_df)
                molecular_descriptor.append(combined_descriptors)
            result_df = pd.concat(molecular_descriptor, ignore_index=True)
            st.info("Loading the model and predicting the emission wavelength, please wait...")
            predictor = TabularPredictor.load("ag-20241119_124834")
            model_options = [
                "WeightedEnsemble_L2", "CatBoost_BAG_L1", "LightGBMLarge_BAG_L1", "LightGBM_BAG_L1", "LightGBMXT_BAG_L1", "NeuralNetTorch_BAG_L1"
            ]
            predictions_dict = {}
            for model in model_options:
                predictions = predictor.predict(result_df, model=model)
                predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")
            # 在结果中标注 WeightedEnsemble_L2 是融合模型
            st.write("Prediction results from various models:")
            st.markdown("**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
            results_df = pd.DataFrame(predictions_dict)
            st.dataframe(results_df)
        except Exception as e:
            st.error(f"An error occurred during molecular descriptor calculation or prediction: {e}")
