from setuptools import setup, find_packages

setup(
    name='dat_analysis',
    version='2.0',
    packages=find_packages('dat_analysis'),
    package_dir={'': 'src'},
    url='https://github.com/TimChild/dat_analysis',
    license='MIT',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Python Analysis Package for Folk Lab at UBC'
)
