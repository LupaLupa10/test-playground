# --- auto-generate requirements.txt from explicit imports in this file ---
import sys
import os
import ast
import importlib

def write_requirements(path="requirements.txt"):
    # Read *this* file
    with open(__file__, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=__file__)

    imports = set()

    # Collect explicit imports only
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    # Standard library modules to skip
    stdlib = {
        "sys", "os", "ast", "importlib", "typing", "pathlib",
        "logging", "json", "math", "time", "re", "datetime"
    }

    lines = []

    for pkg in sorted(imports):
        if pkg in stdlib:
            continue
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", None)
            if ver:
                lines.append(f"{pkg}=={ver}")
        except Exception:
            pass

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"Wrote {len(lines)} packages to {os.path.abspath(path)}")

if __name__ == "__main__":
    write_requirements("requirements.txt")
