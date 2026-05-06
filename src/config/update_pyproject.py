import tomllib

REQ_FILE = "requirements.txt"
PYPROJECT_FILE = "pyproject.toml"


def parse_requirements():
    deps = {}
    with open(REQ_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "==" in line:
                name, version = line.split("==", 1)
                deps[name.lower()] = version
            else:
                deps[line.lower()] = None

    return deps


def parse_pyproject():
    with open(PYPROJECT_FILE, "rb") as f:
        data = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])

    parsed = {}
    for dep in deps:
        if "==" in dep:
            name, version = dep.split("==", 1)
            parsed[name.lower()] = version
        else:
            parsed[dep.lower()] = None

    return data, parsed


def update_pyproject(data, new_deps):
    deps = data.setdefault("project", {}).setdefault("dependencies", [])

    existing_names = {d.split("==")[0].lower() for d in deps}

    for name, version in new_deps.items():
        if name not in existing_names:
            if version:
                deps.append(f"{name}=={version}")
            else:
                deps.append(name)

    return data


def main():
    req_deps = parse_requirements()
    data, py_deps = parse_pyproject()

    missing = {k: v for k, v in req_deps.items() if k not in py_deps}

    if not missing:
        print("All dependencies are already synced.")
        return

    print("Adding missing dependencies:", missing)

    # updated = update_pyproject(data, missing)

    # with open(PYPROJECT_FILE, "wb") as f:
    #    tomli_w.dump(updated, f)


if __name__ == "__main__":
    main()
