import setuptools
import os

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fr:
    requirements = fr.read().splitlines()

setuptools.setup(
    name="demosaic",
    version="v1.0",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@gmail.com",
    maintainer_email="ev.gkanias@gmail.com",
    description="A package providing demosaicing functions for Bayer and polarisation filter patterns.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/demosaic/",
    license="GPLv3+",
    project_urls={
        "Bug Tracker": "https://github.com/evgkanias/demosaic/issues",
        "Source": "https://github.com/evgkanias/demosaic"
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={'': [
        "README.md",
    ]},
    install_requires=requirements,
    python_requires=">=3.10",
)