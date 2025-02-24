import yaml

def convert_yaml_to_requirements(yaml_file, requirements_file):
    # Load the environment.yaml file
    with open(yaml_file, 'r') as f:
        env_data = yaml.safe_load(f)
    
    # Extract the dependencies section
    dependencies = env_data.get('dependencies', [])
    
    # Open the requirements.txt file to write the results
    with open(requirements_file, 'w') as req_file:
        seen = set()  # To handle duplicates
        for dep in dependencies:
            if isinstance(dep, str):
                # Handle dependencies listed as strings (package==version format)
                package_parts = dep.split("=")
                package_name = package_parts[0]
                version = package_parts[1] if len(package_parts) > 1 else ""
                
                if package_name == "pip":  # Skip empty pip entries
                    continue

                # Check if this package has already been written
                if package_name not in seen:
                    req_file.write(f"{package_name}=={version}\n")
                    seen.add(package_name)
            elif isinstance(dep, dict):
                # Handle dependencies specified as dictionaries (e.g., channels)
                for package, version in dep.items():
                    package_name = package
                    version = version.split('=')[0] if isinstance(version, str) else ""
                    
                    if package_name == "pip":  # Skip empty pip entries
                        continue
                    
                    if package_name not in seen:
                        req_file.write(f"{package_name}=={version}\n")
                        seen.add(package_name)
            elif isinstance(dep, list):
                # Handle cases like nested dependencies
                for subdep in dep:
                    package_parts = subdep.split("=")
                    package_name = package_parts[0]
                    version = package_parts[1] if len(package_parts) > 1 else ""
                    
                    if package_name == "pip":  # Skip empty pip entries
                        continue
                    
                    if package_name not in seen:
                        req_file.write(f"{package_name}=={version}\n")
                        seen.add(package_name)

# Example usage
convert_yaml_to_requirements('../../environment.yaml', 'requirements.txt')
