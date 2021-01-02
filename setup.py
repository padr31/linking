import io
import os

from setuptools import find_packages, setup

VERSION = None
with io.open(
    os.path.join(os.path.dirname(__file__), "linking/__init__.py"),
    encoding="utf-8",
) as f:
    for l in f:
        if not l.startswith("__version__"):
            continue
        VERSION = l.split("=")[1].strip(" \"'\n")
        break
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append("setuptools")

setup(
    name="linking",
    version=VERSION,
    description="Machine Learning Library Extensions",
    author="Pavol Drotar",
    author_email="pavol.drotar3@gmail.com",
    url="https://github.com/padr31/linking/",
    packages=find_packages(),
    package_data={"": ["LICENSE.txt", "README.md", "requirements.txt"]},
    include_package_data=True,
    install_requires=install_reqs,
    license="MIT",
    platforms="any",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    long_description="""
Structure-aware molecule generation
""",
)
