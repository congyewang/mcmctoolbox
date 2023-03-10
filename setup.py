from setuptools import setup


setup(
    name='mcmctoolbox',
    version='0.71',
    description='MH, HMC, MALA, tMALA, tMALA/c, KSD',
    url='https://github.com/congyewang/mcmctoolbox',
    author='Congye Wang',
    license='GPLv3+',
    packages=['mcmctoolbox'],
    install_requires=['cvxopt', 'numpy', 'matplotlib', 'stein_thinning']
    )