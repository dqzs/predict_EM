import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile

# 添加 CSS 样式以创建圆角边框
st.markdown(
    """
    <style>
    div.row-widget.stRadio > div{ 
        border-radius: 10px; /* 设置圆角边框的半径 */
        background-color: #f9f9f9; /* 设置背景颜色 */
        padding: 20px; /* 设置内边距 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Predict Fluorescence Emission Wavelengths</h1>
        <blockquote style='margin: auto; width: 90%; background: #f9f9f9; border-left: 0px solid #ccc; padding: 10px; font-size: 1.1em;'>
             The site aims to quickly predict the λem of a molecule based on its structure (SMILES or SDF file) using molecular descriptors and machine learning models. It is recommended to use ChemDraw software to draw the molecule and convert it to sdf.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# 提供两种输入方式
input_option = st.radio("Choose input method:", ("SMILES Input", "SDF File Upload"))
mols = []  # List to store processed molecules

# **SMILES 输入**
if input_option == "SMILES Input":
    smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., CCO")
    if smiles:
        try:
            st.info("Processing SMILES input...")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Convert to 3D molecule
                mol = AllChem.AddHs(mol)
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Use ETKDG algorithm
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

            # Load the single molecule from the SDF file using RDKit
            supplier = Chem.SDMolSupplier(temp_filename)
            for mol in supplier:
                if mol is not None:
                    mols.append(mol)
                    break  # Since we assume only one molecule per file

            if len(mols) > 0:
                st.success("File uploaded successfully, containing 1 valid molecule!")
            else:
                st.error("No valid molecule found in the SDF file!")
        except Exception as e:
            st.error(f"An error occurred while processing the SDF file: {e}")

# Add submit button
submit_button = st.button("Submit and Predict", key="predict_button")

# If submit button is clicked and there are valid molecules
if submit_button and mols:
    with st.spinner("Calculating molecular descriptors and making predictions..."):
        try:
            # Display molecular weights
            st.info("Calculating molecular weights and descriptors...")
            molecular_descriptor = []
            for i, mol in enumerate(mols):
                if mol is None:
                    continue

                # Display molecular weight
                mol_weight = Descriptors.MolWt(mol)
                st.write(f"Molecule {i + 1} Molecular Weight: {mol_weight:.2f} g/mol")

            # Calculate molecular descriptors
            st.info("Calculating molecular descriptors, please wait...")
            calc = Calculator(descriptors, ignore_3D=True)
            mordred_description = []
            rdkit_description = [x[0] for x in Descriptors._descList]
            
            # Compare and filter descriptors
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

            # Combine all molecule descriptor dataframes
            result_df = pd.concat(molecular_descriptor, ignore_index=True)
            result_df = result_df.drop(labels=result_df.dtypes[result_df.dtypes == "object"].index, axis=1)

            # Load AutoGluon model
            st.info("Loading model and making predictions, please wait...")
            predictor = TabularPredictor.load("ag-20241119_124834")

            # Define all model names
            model_options = [
                "LightGBM_BAG_L1", "LightGBMXT_BAG_L1", "CatBoost_BAG_L1",
                "NeuralNetTorch_BAG_L1", "LightGBMLarge_BAG_L1", "WeightedEnsemble_L2"
            ]

            # Store prediction results for each model
            predictions_dict = {}

            for model in model_options:
                predictions = predictor.predict(result_df, model=model)
                predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")  # Add "nm" unit

            # Display prediction results from all models
            st.write("Prediction results from all models:")
            results_df = pd.DataFrame(predictions_dict)
            results_df["Molecule Index"] = range(len(mols))
            results_df = results_df[["Molecule Index"] + model_options]
            st.dataframe(results_df)

        except Exception as e:
            st.error(f"An error occurred during molecular descriptor calculation or prediction: {e}")
