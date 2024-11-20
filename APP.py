import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
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
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # 使用 ETKDG 算法
                if result == -1:
                    st.error("无法生成 3D 构象，请检查分子结构。")
                else:
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

            # 只取第一个有效分子（强行处理为一个分子）
            mol = None
            for molecule in supplier:
                if molecule is not None:
                    mol = molecule
                    break  # 强行选择第一个有效分子
            if mol:
                mols.append(mol)
                st.success("文件上传成功，已选择一个有效分子！")
            else:
                st.error("SDF 文件中没有有效分子！")
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

        # 计算分子描述符
        st.info("正在计算分子描述符，请稍候...")
        calc = Calculator(descriptors, ignore_3D=True)
        mordred_description = []
        rdkit_description = [x[0] for x in Descriptors._descList]
        
        # 比较和筛选描述符
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

        # 合并所有分子的描述符数据框
        result_df = pd.concat(molecular_descriptor, ignore_index=True)
        result_df = result_df.drop(labels=result_df.dtypes[result_df.dtypes == "object"].index, axis=1)

        # 加载 AutoGluon 模型并预测
        st.info("加载模型并进行预测，请稍候...")
        predictor = TabularPredictor.load("ag-20241119_124834")
        predictions = predictor.predict(result_df)

        # 打印所有模型的预测
        model_names = [
            "LightGBM_BAG_L1", "LightGBMXT_BAG_L1", "CatBoost_BAG_L1",
            "NeuralNetTorch_BAG_L1", "LightGBMLarge_BAG_L1", "WeightedEnsemble_L2"
        ]
        
        for model_name in model_names:
            model_prediction = predictor.leaderboard(result_df, silent=True).loc[
                predictor.leaderboard(result_df, silent=True)['model'] == model_name, 'score'
            ]
            if not model_prediction.empty:
                st.write(f"{model_name} 预测结果: {model_prediction.values[0]} nm")
            else:
                st.write(f"{model_name} 没有可用的预测结果。")

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
