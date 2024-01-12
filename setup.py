from setuptools import setup


setup(
    name='mcmctoolbox',
    version='0.15.4',
    description='MH, HMC, MALA, tMALA, tMALA/c, SA, Fisher Adaptive MALA, AM, KSD',
    url='https://github.com/congyewang/mcmctoolbox',
    author='Congye Wang',
    license='GPLv3+',
    packages=['mcmctoolbox'],
    install_requires=['matplotlib', 'numpy', 'scipy', 'seaborn', 'stein_thinning']
    )