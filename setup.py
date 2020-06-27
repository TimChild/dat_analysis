from setuptools import setup

setup(
    name='PyDatAnalysis',
    version='1.0',
    packages=['src', 'tests', 'tests.unit', 'tests.integration'],
    url='https://github.com/TimChild/PyDatAnalysis',
    license='Private',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Python Analysis Package for Folk Lab at UBC',
    long_description=open('README.md').read()
)
