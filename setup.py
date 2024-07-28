from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="virtualhome_eval",
    version="0.1.0",
    author="stanford",
    description="Embodied agent interface evaluation for VirtualHome",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/embodied-agent-eval/virtualhome_eval",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pddlgym",
        "fire",
        "lark",
        "openai",
        "ipdb",
        "networkx>=3.1",
        "numpy>=1.20.0",
    ],
    include_package_data=True,
    package_data={
        "": ["*.json", "*.xml", "*.md", "*.yaml", "*.txt", "*.pddl"],
    },
)

