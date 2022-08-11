from setuptools import setup, find_packages

with open('long_description.txt', 'r') as f:
    long_description = f.read()

setup(
    name='dat_analysis',
    version='3.0.0',
    packages=find_packages('dat_analysis'),
    package_dir={'': 'src'},
    url='https://github.com/TimChild/dat_analysis',
    license='MIT',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Python Analysis Package for Folk Lab at UBC',
    long_description=long_description,
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
        'toml'
    ]
)
