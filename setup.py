from setuptools import setup, find_packages

setup(
    name="speckle",
	package_dir={'':'src'},
	packages=find_packages(where='src'),
    version="0.0.1",
    author="Alex Marek",
    author_email="aleksander.marek.pl@gmail.com",
    description="speckle generation toolbox for digital image correlation",
)