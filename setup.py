import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OmicsMind",
    version="1.0.0",
    author="Zhen Liang",
    author_email="zliang16@ubc.ubc.ca",
    description="Transformer based multi-omics data fill tool",
    packages=["stabl"],
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "torch",
    ],
)
