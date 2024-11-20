import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile

# 页面标题
st.title("预测荧光的发射波长")

# 提供两种输入方式
input_option = st.radio("请选择输入方式：", ("SMILES 输入", "SDF 文件上传"))

mols = []  # 存储处理后的分子

# **SMILES 输入**
if input_option == "SMILES 输入":
    smiles = st.text_input("请输入分子的 SMILES 表示：")
    if smiles:
        try:
            st.info("正在处理 SMILES 输入...")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 转换为 3D 分子
                mol = AllChem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                mols.append(mol)
                st.success("SMILES 转换成功！")
            else:
                st.error("SMILES 输入无效，请检查格式。")
        except Exception as e:
            st.error(f"处理 SMILES 时发生错误: {e}")

# **SDF 文件上传**
elif input_option == "SDF 文件上传":
    uploaded_file = st.file_uploader("请上传一个 SDF 文件", type=["sdf"])
    if uploaded_file:
        try:
            st.info("正在处理 SDF 文件...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filename = temp_file.name

            # 使用 RDKit 加载分子
            supplier = Chem.SDMolSupplier(temp_filename)
            mols += [mol for mol in supplier if mol is not None]

            if len(mols) > 0:
                st.success(f"文件上传成功，共包含 {len(mols)} 个有效分子！")
            else:
                st.error("SDF 文件未包含有效分子！")
        except Exception as e:
            st.error(f"处理 SDF 文件时发生错误: {e}")

# 添加提交按钮
submit_button = st.button("提交并预测")

# 如果点击提交按钮并且存在有效分子
if submit_button and mols:
    try:
        # 显示分子量
        st.info("正在计算分子量和分子描述符...")
        for i, mol in enumerate(mols):
            if mol is None:
                continue

            # 显示分子量
            mol_weight = Descriptors.MolWt(mol)
            st.write(f"分子 {i + 1} 的分子量：{mol_weight:.2f}")

        # 计算指定的分子描述符
        st.info("正在计算分子描述符，请稍候...")
        calc = Calculator(descriptors, ignore_3D=True)
        descriptor_names = ["SdsCH", "MolLogP", "SdssC", "VSA_EState7", 
                            "SlogP_VSA8", "VE1_A", "EState_VSA4", "AATS8i", "AATS4i"]

        Molecular_descriptor = []
        for mol in mols:
            mordred_df = pd.DataFrame(calc.pandas([mol]))
            filtered_df = mordred_df[descriptor_names]
            Molecular_descriptor.append(filtered_df)

        # 合并所有分子的描述符数据框
        result_df = pd.concat(Molecular_descriptor, ignore_index=True)

        # 加载 AutoGluon 模型并预测
        st.info("加载模型并进行预测，请稍候...")
        predictor = TabularPredictor.load("ag-20241119_124834")
        predictions = predictor.predict(result_df)

        # 将预测结果保留为整数
        predictions = predictions.astype(int)

        # 展示结果
        st.write("预测结果：")
        results = pd.DataFrame({
            "分子索引": range(len(mols)),
            "预测发射波长 (nm)": predictions
        })
        st.dataframe(results)

    except Exception as e:
        st.error(f"处理分子描述符或预测时发生错误: {e}")
