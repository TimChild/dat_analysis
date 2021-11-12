from distutils.core import setup

setup(
    name='dat_analysis',
    version='2.0',
    packages=['src', 'src.plotting',
              'src.plotting.mpl', 'src.plotting.plotly', 'src.plotting.plotly.common_plots', 'src.dat_object',
              'src.dat_object.attributes', 'src.analysis_tools', 'src.data_standardize',
              'src.data_standardize.exp_specific', 'tests'],
    url='https://github.com/TimChild/dat_analysis',
    license='MIT',
    author='Tim Child',
    author_email='timjchild@gmail.com',
    description='Python Analysis Package for Folk Lab at UBC'
)
