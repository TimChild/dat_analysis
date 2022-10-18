from setuptools import setup, find_packages

with open('long_description.txt', 'r') as f:
    long_description = f.read()

setup(
    name='dat_analysis',
    version='3.0.1',
    url='https://github.com/TimChild/dat_analysis',
    license='MIT',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Python Analysis Package for Folk Lab at UBC',
    long_description=long_description,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={'dat_analysis': ["*.mat", "*.txt"]},
    python_requires='>=3.10',
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'plotly',
        'h5py',
        'lmfit',
        'dictor',
        'scipy',
        'pillow',
        'deprecation',
        'slugify',
        'igorwriter',
        'singleton_decorator',
        'progressbar2',
        'numdifftools',  # For lmfit uncertainties on powell method
        'filelock',  # For accessing HDFs safely between processes
        'toml',
        'jupyter',  # Only here because it is so often useful in the environment
        'jupyterlab',  # Only here because it is so often useful in the environment
        'progressbar',  # Often useful when working with large datasets
        'opencv-python',  # For shift_tracker_algorithm (import cv2)
    ]
)
