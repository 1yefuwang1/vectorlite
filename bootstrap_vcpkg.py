import platform
import subprocess
import os

system = platform.system().lower()

if system == 'windows':
    subprocess.run([os.path.join('vcpkg', 'bootstrap-vcpkg.bat')], check=True)
else:
    subprocess.run([os.path.join('vcpkg', './bootstrap-vcpkg.sh')], check=True)