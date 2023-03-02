from setuptools import setup


setup(
    name='mcmctoolbox',
    version='0.2',
    description='MH, HMC, MALA, tMALA, tMALA/c',
    url='https://github.com/congyewang/mcmctoolbox',
    author='Congye Wang',
    license='GPLv3+',
    packages=['mcmctoolbox'],
    install_requires=['numpy', 'matplotlib', 'tqdm', 'stein_thinning']
    )