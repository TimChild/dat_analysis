from setuptools import setup, find_packages


setup(
    name='dat_analysis',
    version='2.0',
    # packages=['dat_analysis', 'dat_analysis.plotting',
    #           'dat_analysis.plotting.mpl', 'dat_analysis.plotting.plotly', 'dat_analysis.plotting.plotly.common_plots', 'dat_analysis.dat_object',
    #           'dat_analysis.dat_object.attributes', 'dat_analysis.analysis_tools', 'dat_analysis.data_standardize',
    #           'dat_analysis.data_standardize.exp_specific', 'tests'],
    packages=find_packages('dat_analysis'),
    package_dir={'': 'dat_analysis'},
    url='https://github.com/TimChild/dat_analysis',
    license='MIT',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Python Analysis Package for Folk Lab at UBC'
)
