from typing import List

import subprocess

# Get the list of installed packages
result: subprocess.CompletedProcess = subprocess.run(
    ["pip", "list", "--format=freeze"],
    stdout=subprocess.PIPE
)
packages: List[str] = result.stdout.decode("utf-8").split("\n")

# Remove versions
package_names: List[str] = [
    each_pkg.split("==")[0]
    for each_pkg in packages
]

# Write to requirements-no-versions.txt
with open("requirements-no-versions.txt", "w+") as f:
    for each_pkg in package_names:
        f.write(f"{each_pkg}\n")