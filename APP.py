import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile

# 添加 CSS 样式，定义更紧凑的圆角框
st.markdown(
    """
    <style>
    .rounded-container {
        border: 2px solid #000000; /* 黑色边框 */
        border-radius: 20px; /* 圆角边框 */
        padding: 15px; /* 内边距 */
        margin: 10px auto; /* 外边距，居中 */
        max-width: 80%; /* 最大宽度，适配窗口 */
        background-color: #f9f9f9; /* 背景颜色 */
    }
    .stApp {
        background-color: #ffffff; /* 页面背景 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 包裹所有内容的 div
st.markdown("<div class='rounded-container'>", unsafe_allow_html=True)

# 页面标题和简介
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Predict Originc Fluorescence Emission Wavelengths</h1>
        <blockquote style='margin: auto; background: #f9f9f9; border-left: 4px solid #ccc; padding: 10px; font-size: 1.1em; max-width: 90%;'>
            This website aims to quickly predict the emission wavelength of a molecule based on its structure (SMILES or SDF file) using machine learning models. It is recommended to use ChemDraw software to draw molecules and convert them to sdf.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 提供两种输入方式
input_option = st.radio("Choose input method:", ("SMILES Input", "SDF File Upload"))
mols = []  # 存储分子列表

# SMILES 输入
if input_option == "SMILES Input":
    smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., NC1=CC=C(C=C1)C(=O)O")
    if smiles:
        try:
            st.info("Processing SMILES input...")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 转换为3D分子
                AllChem.AddHs(mol)
                AllChem.EmbedMolecule(mol)  # 使用 ETKDG 算法
                mols.append(mol)
            else:
                st.error("Invalid SMILES input. Please check the format.")
        except Exception as e:
            st.error(f"An error occurred while processing SMILES: {e}")

# SDF 文件上传
elif input_option == "SDF File Upload":
    uploaded_file = st.file_uploader("Upload an SDF file", type=["sdf"])
    if uploaded_file:
        try:
            st.info("Processing SDF file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filename = temp_file.name

            # 使用 RDKit 加载单个分子
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

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 如果点击提交按钮且存在有效分子
if submit_button and mols:
    with st.spinner("Calculating molecular descriptors and making predictions..."):
        try:
            # 显示分子量
            st.info("Calculating molecular weights and descriptors...")
            molecular_descriptor = []
            for i, mol in enumerate(mols):
                if mol is None:
                    continue

                # 显示分子量
                mol_weight = Descriptors.MolWt(mol)
                st.write(f"Molecule {i + 1} Molecular Weight: {mol_weight:.2f} g/mol")

            # 计算分子描述符
            st.info("Calculating molecular descriptors, please wait...")
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_description = []
            rdkit_description = [x[0] for x in Descriptors._descList]

            # 比较并过滤描述符
            for i in calc.descriptors:
                mordred_description.append(i.__str__())
            for i in mordred_description:
                if i in rdkit_description:
                    rdkit_description.remove(i)

            descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_description)

            molecular_descriptor = []
            for mol in mols:
                calculator_descript = pd.DataFrame(calc.pandas([mol]))
                rdkit_descriptors = pd.DataFrame(
                    [descriptor_calculator.CalcDescriptors(mol)],
                    columns=rdkit_description
                )
                combined_descript = calculator_descript.join(rdkit_descriptors)
                molecular_descriptor.append(combined_descript)

            # 合并所有分子描述符数据帧
            result_df = pd.concat(molecular_descriptor, ignore_index=True)
            result_df = result_df.drop(labels=result_df.dtypes[result_df.dtypes == "object"].index, axis=1)

            # 加载 AutoGluon 模型
            st.info("Loading the model and predicting the emission wavelength, please wait...")
            predictor = TabularPredictor.load("ag-20241119_124834")

            # 定义所有模型名称
            model_options = [
                "LightGBM_BAG_L1", "LightGBMXT_BAG_L1", "CatBoost_BAG_L1",
                "NeuralNetTorch_BAG_L1", "LightGBMLarge_BAG_L1", "WeightedEnsemble_L2"
            ]

            # 存储每个模型的预测结果
            predictions_dict = {}

            for model in model_options:
                predictions = predictor.predict(result_df, model=model)
                predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")  # 添加单位

            # 显示所有模型的预测结果
            st.write("Prediction results from all models:")
            results_df = pd.DataFrame(predictions_dict)
            results_df["Molecule Index"] = range(len(mols))
            results_df = results_df[["Molecule Index"] + model_options]
            st.dataframe(results_df)

        except Exception as e:
            st.error(f"An error occurred during molecular descriptor calculation or prediction: {e}")

# 关闭圆角框 div
st.markdown("</div>", unsafe_allow_html=True)
