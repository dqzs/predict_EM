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
st.markdown("<h3 style='color: #FF6347;'>基于分子结构预测发射波长</h3>", unsafe_allow_html=True)

# 提供两种输入方式
input_option = st.radio("请选择输入方式：", ("SMILES 输入", "SDF 文件上传"))

mols = []  # 存储处理后的分子

# **SMILES 输入**
if input_option == "SMILES 输入":
    smiles = st.text_input("请输入分子的 SMILES 表示：", placeholder="例如：CCO")
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
                    if mol not in mols:  # 检查分子是否已经被添加
                        mols.append(mol)
                        st.success("SMILES 转换成功！")
                    else:
                        st.warning("分子已存在，跳过添加。")
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
            num_mols = len(list(supplier))  # 获取分子数量
            st.write(f"SDF 文件包含 {num_mols} 个分子。")  # 打印分子数量

            for mol in supplier:
                if mol is not None and mol not in mols:  # 检查分子是否已经被添加
                    mols.append(mol)
                    #st.write(f"分子 {mol.GetProp('temp_file.name')} 已添加。")  # 打印分子名称

            if len(mols) > 0:
                st.success(f"文件上传成功，共包含 {len(mols)} 个有效分子！")
            else:
                st.error("SDF 文件未包含有效分子！")
        except Exception as e:
            st.error(f"处理 SDF 文件时发生错误: {e}")

# 添加提交按钮
submit_button = st.button("提交并预测", key="predict_button")

# 如果点击提交按钮并且存在有效分子
if submit_button and mols:
    with st.spinner("正在计算分子描述符并进行预测..."):
        try:
            # 显示分子量
            st.info("正在计算分子量和分子描述符...")
            molecular_descriptor = []
            for i, mol in enumerate(mols):
                if mol is None:
                    continue

                # 显示分子量
                mol_weight = Descriptors.MolWt(mol)
                st.write(f"分子 {i + 1} 的分子量：{mol_weight:.2f} g/mol")

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

            # 加载 AutoGluon 模型
            st.info("加载模型并进行预测，请稍候...")
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
                predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")  # 添加 "nm" 单位

            # 展示所有模型的预测结果
            st.write("所有模型的预测结果：")
            results_df = pd.DataFrame(predictions_dict)
            results_df["分子索引"] = range(len(mols))
            results_df = results_df[["分子索引"] + model_options]
            st.dataframe(results_df)

        except Exception as e:
            st.error(f"处理分子描述符或预测时发生错误: {e}")
