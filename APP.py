import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from mordred import Calculator
import pandas as pd

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
        # 指定需要计算的描述符
        required_descriptors = {
            'SdsCH': 'SdssC', 'MolLogP': 'MolLogP', 'VSA_EState7': 'VSA_EState7',
            'SlogP_VSA8': 'SlogP_VSA8', 'VE1_A': 'VE1_A', 'EState_VSA4': 'EState_VSA4',
            'AATS8i': 'AATS8i', 'AATS4i': 'AATS4i'
        }
        calc = Calculator(required_descriptors.keys(), ignore_3D=True)

        # 计算分子描述符
        st.info("正在计算分子描述符，请稍候...")
        Molecular_descriptor = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue

            descriptors_result = calc.pandas([mol])
            if descriptors_result.empty:
                st.error(f"分子 {i + 1} 的描述符计算失败。")
                continue

            descriptors_df = pd.DataFrame(descriptors_result)
            Molecular_descriptor.append(descriptors_df)

        # 合并所有分子的描述符数据框
        result_df = pd.concat(Molecular_descriptor, ignore_index=True)

        # 展示结果
        st.write("预测结果：")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"处理分子描述符或预测时发生错误: {e}")
