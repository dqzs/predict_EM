import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile

# 外层大边框容器
st.markdown(
    """
    <div style='
        border: 2px solid #ccc; 
        border-radius: 15px; 
        padding: 20px; 
        margin: 20px auto; 
        width: 80%; 
        background: #f9f9f9;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'
    >
    """,
    unsafe_allow_html=True,
)

# 页面标题及说明
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='margin: 0 auto;'>Prediction of fluorescence emission λem based on molecular structure</h1>
    </div>
    <div style='margin: 10px auto; text-align: justify; background: #ffffff; border: 1px solid #ddd; padding: 15px; border-radius: 10px;'>
        <p style='font-size: 1.1em; line-height: 1.6;'>
            The tool is designed to quickly predict the λem of a molecule based on its structure (SMILES or SDF file) using molecular descriptors and machine learning models. 
            It is recommended to use sdf files of molecules drawn by ChemDraw software.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 提供两种输入方式并使其居中
st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
input_option = st.radio("Choose input method:", ("SMILES Input", "SDF File Upload"), index=0)
st.markdown("</div>", unsafe_allow_html=True)

# SMILES 输入框居中
if input_option == "SMILES Input":
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    smiles = st.text_input(
        "Enter the SMILES representation of the molecule:",
        placeholder="e.g., CCO",
        key="smiles_input",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# SDF 文件上传居中
elif input_option == "SDF File Upload":
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an SDF file", type=["sdf"], key="sdf_upload")
    st.markdown("</div>", unsafe_allow_html=True)

# 提交按钮居中
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
submit_button = st.button("Submit and Predict", key="predict_button")
st.markdown("</div>", unsafe_allow_html=True)

# 处理和预测功能
mols = []  # List to store processed molecules

# **SMILES 输入**
if input_option == "SMILES Input" and smiles:
    try:
        st.info("Processing SMILES input...")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Convert to 3D molecule
            mol = AllChem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Use ETKDG algorithm
            if result == -1:
                st.error("Failed to generate 3D conformation. Please check the molecular structure.")
            else:
                AllChem.MMFFOptimizeMolecule(mol)
                if mol not in mols:  # Check if molecule is already added
                    mols.append(mol)
                    st.success("SMILES converted successfully!")
                else:
                    st.warning("Molecule already exists. Skipping.")
        else:
            st.error("Invalid SMILES input. Please check the format.")
    except Exception as e:
        st.error(f"An error occurred while processing SMILES: {e}")

# **SDF 文件上传**
elif input_option == "SDF File Upload" and uploaded_file:
    try:
        st.info("Processing SDF file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_filename = temp_file.name

        # Load molecules using RDKit
        supplier = Chem.SDMolSupplier(temp_filename)
        num_mols = len(list(supplier))  # Count molecules
        st.write(f"The SDF file contains {num_mols} molecules.")  # Display molecule count

        for mol in supplier:
            if mol is not None and mol not in mols:  # Check if molecule is already added
                mols.append(mol)

        if len(mols) > 0:
            st.success(f"File uploaded successfully, containing {len(mols)} valid molecules!")
        else:
            st.error("No valid molecules found in the SDF file!")
    except Exception as e:
        st.error(f"An error occurred while processing the SDF file: {e}")

# **提交按钮**
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

# 关闭大边框的外层容器
st.markdown("</div>", unsafe_allow_html=True)
