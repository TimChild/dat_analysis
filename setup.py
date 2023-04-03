from setuptools import setup, find_packages

with open('long_description.txt', 'r') as f:
    long_description = f.read()

setup(
    name='dat_analysis',
    version='3.2.0',
    url='https://github.com/TimChild/dat_analysis',
    license='MIT',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Python Analysis Package for Folk Lab at UBC',
    long_description=long_description,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,  # For including files in MANIFEST.in
    python_requires='>=3.10',
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'plotly',
        'kaleido',  # For showing plotly plots
        'h5py',
        'lmfit',
        'dictor',
        'scipy',
        'pillow',
        'deprecation',
        'python-slugify',
        'igorwriter',
        'singleton_decorator',
        'numdifftools',  # For lmfit uncertainties on powell method
        'filelock',  # For accessing HDFs safely between processes
        'toml',
        'jupyter',  # Only here because it is so often useful in the environment
        'jupyter_contrib_nbextensions',  # For exporting jupyter notebooks to pdf
        'jupyterlab',  # Only here because it is so often useful in the environment
        'tdqm',  # Progressbar often useful when working with large datasets
        'opencv-python',  # For shift_tracker_algorithm (import cv2)
        'nb_black',  # For formatting jupyter files with %load_ext lab_black
        'dash>2.0',  # For making interactive dash apps
        'jupyter-dash',  # Enables dash apps to work in jupyter without blocking cells
        'nodejs-bin[cmd]',  # Node>14 required for 'jupyter lab build' after installing jupyter-dash
        'dash-bootstrap-components',
        'dash-extensions',  # Provides ServersideOutput among many other things
        'tabulate',  # Required by dash or dash-bootstrap-components
    ]
)
