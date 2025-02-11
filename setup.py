from setuptools import setup, find_packages


setup(
    name='soso',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'basemap',
        'intervaltree',
        'matplotlib',
        'networkx',
        'numpy',
        'pandas',
        'skyfield',
        'ortools'
    ],
    entry_points={},
    description='Satellite operations services optimizer.',
    url='https://github.com/ENG4000-SOSO/genetic-algorithms',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
