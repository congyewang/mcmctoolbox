from setuptools import setup


setup(
    name='mcmctoolbox',
    version='0.10',
    description='MH, HMC, MALA, tMALA, tMALA/c, KSD',
    url='https://github.com/congyewang/mcmctoolbox',
    author='Congye Wang',
    license='GPLv3+',
    packages=['mcmctoolbox'],
    install_requires=['matplotlib', 'numpy', 'scipy', 'stein_thinning']
    )