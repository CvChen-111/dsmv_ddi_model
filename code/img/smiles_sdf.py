import os
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

# ================= 配置区 =================
CSV_PATH = "/data/cclsol/cfn/MF-Qwen/CD/vision/572drug_smile.csv"
OUT_DIR = "drug_3d"
NUM_CONFS = 1          # 每个分子生成多少个构象（先 5 个就够）
RANDOM_SEED = 42
# =========================================

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

success, failed = 0, 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    drug_id = row["drug_id"]
    name = row["name"]
    smiles = row["smiles"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[FAIL] SMILES parse error: {drug_id}")
        failed += 1
        continue

    # 1. 加氢
    mol = Chem.AddHs(mol)

    # 2. 3D 构象生成（ETKDG）
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.numThreads = 0

    try:
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=NUM_CONFS,
            params=params
        )
    except Exception as e:
        print(f"[FAIL] Embed error: {drug_id}")
        failed += 1
        continue

    if len(conf_ids) == 0:
        print(f"[FAIL] No conformers: {drug_id}")
        failed += 1
        continue

    # 3. 力场优化（MMFF 优先）
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except:
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol)
        except:
            print(f"[WARN] Optimization failed: {drug_id}")

    # 4. 保存为 SDF（一个文件一个分子，包含多个 conformers）
    out_path = os.path.join(OUT_DIR, f"{drug_id}.sdf")
    writer = Chem.SDWriter(out_path)
    for conf_id in conf_ids:
        writer.write(mol, confId=conf_id)
    writer.close()

    success += 1

print(f"\nDone! Success: {success}, Failed: {failed}")
