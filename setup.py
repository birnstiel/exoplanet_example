"""
Setup file for package `exoplanet_example`.
"""
from setuptools import setup
import pathlib

PACKAGENAME = 'exoplanet_example'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).absolute().parent


def read_version():
    with (HERE / PACKAGENAME / '__init__.py').open() as fid:
        for line in fid:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":

    setup(
        name=PACKAGENAME,
        description='some exoplanet notebooks to be shared with voila',
        version=read_version(),
        long_description=(HERE / "Readme.rst").read_text(),
        long_description_content_type='text/x-rst',
        url='https://github.com/birnstiel/exoplanet_example',
        author='Til Birnstiel',
        author_email='til.birnstiel@lmu.de',
        license='GPLv3',
        packages=[PACKAGENAME],
        package_dir={PACKAGENAME: PACKAGENAME},
        package_data={PACKAGENAME: [
            'data/*.*',
            'notebooks/*.ipynb',
        ]},
        install_requires=[
            'numpy',
            'matplotlib',
            'rebound',
            'bqplot',
            'ipympl',
            'scipy',
            'voila',
            'ipywidgets',
            'batman-package',
        ],
        zip_safe=True,
    )
