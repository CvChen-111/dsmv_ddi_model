"""Render optimized 3D molecular conformers into four-view PNG images."""
import os
from rdkit import Chem
import py3Dmol
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from io import BytesIO
from PIL import Image

# ---------------- Configuration ----------------
SDF_DIR = "drug_3d"  # SDF directory
OUTPUT_DIR = "drug_3d_png"  # Output PNG directory
VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0)]  # Four rotation views

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Headless Chrome configuration ----------------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--window-size=800,800")
driver = webdriver.Chrome(options=chrome_options)


# ---------------- Rendering function ----------------
def render_sdf_to_png(sdf_path, output_prefix):
    """Render one SDF molecule from four viewpoints and save PNG files."""
    mol_supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol_supplier = [m for m in mol_supplier if m is not None]
    if len(mol_supplier) == 0:
        print(f"[WARN] No valid molecule in {sdf_path}")
        return
    mol = mol_supplier[0]

    # Convert the molecule to a molblock with 3D coordinates.
    mol_block = Chem.MolToMolBlock(mol)

    # Create a py3Dmol viewer.
    for idx, (rot_y, rot_x) in enumerate(VIEWS):
        view = py3Dmol.view(width=400, height=400)
        view.addModel(mol_block, "mol")
        view.setStyle({'stick': {}})
        view.zoomTo()
        view.rotate(rot_x, rot_y)

        # Decode the base64 image returned by py3Dmol.
        img_b64 = view.png()
        img_data = base64.b64decode(img_b64.split(',')[1])
        img = Image.open(BytesIO(img_data))

        # Save the rendered view.
        out_path = f"{output_prefix}_view{idx}.png"
        img.save(out_path)
        print(f"[INFO] Saved {out_path}")


# ---------------- Batch processing ----------------
for sdf_file in os.listdir(SDF_DIR):
    if sdf_file.endswith(".sdf"):
        sdf_path = os.path.join(SDF_DIR, sdf_file)
        drug_id = sdf_file.split(".")[0]
        output_prefix = os.path.join(OUTPUT_DIR, drug_id)
        render_sdf_to_png(sdf_path, output_prefix)

driver.quit()
print("All done!")
