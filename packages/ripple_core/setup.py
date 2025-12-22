from setuptools import setup, find_packages

setup(
    name="ripple-core",
    version="0.1.0",
    description="Reusable ripple analysis library",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy", "scipy", "pandas", "mne", "pywavelets", "h5py", "xarray", "joblib"
    ],
)
