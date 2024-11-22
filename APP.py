import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile

# 自定义CSS样式
CUSTOM_CSS = """
    <style>
        .stApp {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 90%;
            margin: auto;
            overflow: hidden;
            background-color: #fff;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        p {
            text-align: justify;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .stError {
            color: red;
        }
        .stWarning {
            color: orange;
        }
        .stSuccess {
            color: green;
        }
        .stInfo {
            color: blue;
        }
    </style>
"""

# 将自定义CSS添加到Streamlit页面
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 页面标题及说明
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='margin: 0 auto;'>Prediction of fluorescence emission λem based on molecular structure</h1>
    </div>
    <div style='margin: 10px auto; text-align: justify; background: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 5px;'>
        <p style='font-size: 1.1em; line-height: 1.6;'>
            The tool is designed to quickly predict the λem of a molecule based on its structure (SMILES or SDF file) using molecular descriptors and machine learning models. 
            It is recommended to use sdf files of molecules drawn by ChemDraw software.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 提供两种输入方式
input_option = st.radio("Choose input method:", ("SMILES Input", "SDF File Upload"), index=0)

# 设置分子存储列表
mols = []

# **SMILES 输入**
if input_option == "SMILES Input":
    smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., CCO")
    if smiles:
        try:
            st.info("Processing SMILES input...")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = AllChem.AddHs(mol)
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())  
                if result == -1:
                    st.error("Failed to generate 3D conformation. Please check the molecular structure.")
                else:
                    AllChem.MMFFOptimizeMolecule(mol)
                    if mol not in mols:
                        mols.append(mol)
                        st.success("SMILES converted successfully!")
                    else:
                        st.warning("Molecule already exists. Skipping.")
            else:
                st.error("Invalid SMILES input. Please check the format.")
        except Exception as e:
            st.error(f"An error occurred while processing SMILES: {e}")

# **SDF 文件上传**
elif input_option == "SDF File Upload":
    uploaded_file = st.file_uploader("Upload an SDF file", type=["sdf"])
    if uploaded_file:
        try:
            st.info("Processing SDF file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filename = temp_file.name

            supplier = Chem.SDMolSupplier(temp_filename)
            for mol in supplier:
                if mol is not None and mol not in mols:
                    mols.append(mol)

            if len(mols) > 0:
                st.success(f"File uploaded successfully, containing {len(mols)} valid molecules!")
            else:
                st.error("No valid molecules found in the SDF file!")
        except Exception as e:
            st.error(f"An error occurred while processing the SDF file: {e}")

# 添加提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 如果提交有效分子
if submit_button and mols:
    with st.spinner("Calculating molecular descriptors and making predictions..."):
        try:
            st.info("Calculating molecular weights and descriptors...")
            molecular_descriptor = []
            for i, mol in enumerate(mols):
                if mol is None:
                    continue
                mol_weight = Descriptors.MolWt(mol)
                st.write(f"Molecule {i + 1} Molecular Weight: {mol_weight:.2f} g/mol")

            st.info("Calculating molecular descriptors, please wait...")
            calc = Calculator(descriptors, ignore_3D=True)
            descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

            molecular_descriptor = []
            for mol in mols:
                mordred_descriptors = pd.DataFrame(calc.pandas([mol]))
                rdkit_descriptors = pd.DataFrame(
                    [descriptor_calculator.CalcDescriptors(mol)],
                    columns=descriptor_calculator.GetDescriptorNames()
                )
                combined_descriptors = mordred_descriptors.join(rdkit_descriptors)
                molecular_descriptor.append(combined_descriptors)

            result_df = pd.concat(molecular_descriptor, ignore_index=True)
            result_df = result_df.dropna(axis=1, how="any")

            st.info("Loading model and making predictions, please wait...")
            predictor = TabularPredictor.load("ag-20241119_124834")
            model_options = [
                "LightGBM_BAG_L1", "LightGBMXT_BAG_L1", "CatBoost_BAG_L1",
                "NeuralNetTorch_BAG_L1", "LightGBMLarge_BAG_L1", "WeightedEnsemble_L2"
            ]

            predictions_dict = {}
            for model in model_options:
                predictions = predictor.predict(result_df, model=model)
                predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")

            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.write("Prediction results from all models:")
            results_df = pd.DataFrame(predictions_dict)
            results_df["Molecule Index"] = range(len(mols))
            st.dataframe(results_df.style.set_table_styles(
                [{'selector': 'thead th', 'props': 'text-align: center;'},
                 {'selector': 'tbody td', 'props': 'text-align: center;'}]
            ))
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during molecular descriptor calculation or prediction: {e}")
