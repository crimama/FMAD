"""Patch AnomalyDINO to add MVTecAD2 dataset support in all relevant files."""
import sys
from pathlib import Path

repo = Path(sys.argv[1])
AD2_OBJECTS = '["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]'

# 1. Patch src/utils.py - get_dataset_info
utils_path = repo / "src" / "utils.py"
code = utils_path.read_text()
if "MVTecAD2" not in code:
    patch = f'''
    elif dataset == "MVTecAD2":
        objects = {AD2_OBJECTS}
        object_anomalies = {{obj: ["bad"] for obj in objects}}
        masking_default = {{obj: False for obj in objects}}
        rotation_default = {{obj: False for obj in objects}}
        return objects, object_anomalies, masking_default, rotation_default
'''
    marker = 'elif dataset == "VisA"'
    idx = code.find(marker)
    line_start = code.rfind('\n', 0, idx) + 1
    code = code[:line_start] + patch + '\n' + code[line_start:]
    utils_path.write_text(code)
    print("Patched src/utils.py")

# 2. Patch src/post_eval.py - get_objects_from_dataset
post_eval_path = repo / "src" / "post_eval.py"
code = post_eval_path.read_text()
if "MVTecAD2" not in code:
    old = '    return objects'
    # Find the return in get_objects_from_dataset (last 'return objects' after VisA block)
    idx = code.find('def get_objects_from_dataset')
    ret_idx = code.find('    return objects', idx)
    new = f'''    elif dataset == "MVTecAD2":
        objects = {AD2_OBJECTS}
    return objects'''
    code = code[:ret_idx] + new + code[ret_idx + len(old):]
    post_eval_path.write_text(code)
    print("Patched src/post_eval.py")

# 3. Patch src/post_eval.py - parse_dataset_files to treat MVTecAD2 like MVTec
# Only patch inside parse_dataset_files, not get_objects_from_dataset
code = post_eval_path.read_text()
func_start = code.find('def parse_dataset_files')
func_end = code.find('\ndef ', func_start + 1)
func_body = code[func_start:func_end]
func_body_new = func_body.replace(
    'dataset == "MVTec"',
    'dataset in ("MVTec", "MVTecAD2")'
)
code = code[:func_start] + func_body_new + code[func_end:]
post_eval_path.write_text(code)
print("Patched src/post_eval.py parse_dataset_files")

print("All patches applied successfully")
