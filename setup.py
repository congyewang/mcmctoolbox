from setuptools import setup


setup(
    name='mcmctoolbox',
    version='0.11.2',
    description='MH, HMC, MALA, tMALA, tMALA/c, Q-invariant MALA, KSD',
    url='https://github.com/congyewang/mcmctoolbox',
    author='Congye Wang',
    license='GPLv3+',
    packages=['mcmctoolbox'],
    install_requires=['matplotlib', 'numpy', 'scipy', 'seaborn', 'stein_thinning']
    )