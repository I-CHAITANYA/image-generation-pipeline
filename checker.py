import pkg_resources
import sys
import re

def parse_requirements(file_path):
    packages = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove inline comments and version specifiers
                package_name = re.split('[=<>!~;@]', line)[0].strip()
                if package_name:
                    packages.append(package_name)
    return packages

def main():
    req_file = 'requirements.txt'
    required = parse_requirements(req_file)
    installed = {pkg.key for pkg in pkg_resources.working_set}

    missing = []
    for pkg in required:
        if pkg.lower() not in installed:
            missing.append(pkg)

    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
    else:
        print("✅ All packages from requirements.txt are installed!")

if __name__ == "__main__":
    main()