"""Apply the two upstream installer fixes, with exact version/content checks."""
from importlib.metadata import distribution, version
from pathlib import Path


def replace_checked(path, before, after):
    text = path.read_text()
    if after in text and before not in text:
        print(f"already patched: {path}")
        return
    if text.count(before) != 1:
        raise RuntimeError(f"Unexpected source in {path}; refusing to patch")
    path.write_text(text.replace(before, after))
    print(f"patched: {path}")


assert version("sapien") == "3.0.0b1"
assert version("mplib") == "0.2.1"
replace_checked(
    Path(distribution("mplib").locate_file("mplib/planner.py")),
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
)
p = Path(distribution("sapien").locate_file("sapien/wrapper/urdf_loader.py"))
# Official installer changes the URDF reader encoding; retain all other semantics.
replace_checked(p, 'with open(urdf_file, "r") as f:',
                'with open(urdf_file, "r", encoding="utf-8") as f:')
