"""Patch AnomalyDINO for RobustAD (MetalParts only, has masks)."""
import sys
from pathlib import Path

repo = Path(sys.argv[1])
OBJECTS = "['MetalParts_test0','MetalParts_test1','MetalParts_test2','MetalParts_test3','MetalParts_test4','MetalParts_test5','MetalParts_test6']"

# 1. Patch src/utils.py
utils_path = repo / "src" / "utils.py"
code = utils_path.read_text()
if "RobustAD" not in code:
    patch = f'''
    elif dataset == "RobustAD":
        objects = {OBJECTS}
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

# 2. Patch src/post_eval.py
pe_path = repo / "src" / "post_eval.py"
code = pe_path.read_text()
if "RobustAD" not in code:
    old = '    return objects'
    idx = code.find('def get_objects_from_dataset')
    ret_idx = code.find('    return objects', idx)
    new = f'''    elif dataset == "RobustAD":
        objects = {OBJECTS}
    return objects'''
    code = code[:ret_idx] + new + code[ret_idx + len(old):]
    pe_path.write_text(code)
    print("Patched post_eval.py get_objects")

# 3. Patch parse_dataset_files for RobustAD
code = pe_path.read_text()
func_start = code.find('def parse_dataset_files')
func_end = code.find('\ndef ', func_start + 1)
func = code[func_start:func_end]
if '"RobustAD"' not in func:
    func = func.replace('dataset == "MVTec"', 'dataset in ("MVTec", "RobustAD")')
    code = code[:func_start] + func + code[func_end:]
    pe_path.write_text(code)
    print("Patched post_eval.py parse_dataset_files")

print("Done")
