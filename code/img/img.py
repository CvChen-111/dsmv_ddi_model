import os
from rdkit import Chem
import py3Dmol
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from io import BytesIO
from PIL import Image

# ---------------- 配置 ----------------
SDF_DIR = "drug_3d"  # SDF 文件夹
OUTPUT_DIR = "drug_3d_png"  # 输出 PNG 文件夹
VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0)]  # 4个旋转视角

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- headless Chrome 配置 ----------------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--window-size=800,800")
driver = webdriver.Chrome(options=chrome_options)


# ---------------- 渲染函数 ----------------
def render_sdf_to_png(sdf_path, output_prefix):
    mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol_supplier = [m for m in mol_supplier if m is not None]
    if len(mol_supplier) == 0:
        print(f"[WARN] No valid molecule in {sdf_path}")
        return
    mol = mol_supplier[0]

    # 转换为 molblock (3D 坐标)
    mol_block = Chem.MolToMolBlock(mol)

    # 创建 py3Dmol Viewer
    for idx, (rot_y, rot_x) in enumerate(VIEWS):
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mol_block, "mol")
        view.setStyle({'stick': {}})
        view.zoomTo()
        view.rotate(rot_x, rot_y)

        # 获取 base64 图片
        img_b64 = view.png()
        img_data = base64.b64decode(img_b64.split(',')[1])
        img = Image.open(BytesIO(img_data))

        # 保存
        out_path = f"{output_prefix}_view{idx}.png"
        img.save(out_path)
        print(f"[INFO] Saved {out_path}")


# ---------------- 批量处理 ----------------
for sdf_file in os.listdir(SDF_DIR):
    if sdf_file.endswith(".sdf"):
        sdf_path = os.path.join(SDF_DIR, sdf_file)
        drug_id = sdf_file.split(".")[0]
        output_prefix = os.path.join(OUTPUT_DIR, drug_id)
        render_sdf_to_png(sdf_path, output_prefix)

driver.quit()
print("All done!")
