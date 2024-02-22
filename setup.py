from setuptools import setup, find_packages

def read_file(fpath):
    with open(fpath) as fp:
        data = fp.read()
    return data

setup(
    name="whalegrad",
    version="0.0.2",
    author="Saurabh Aone",
    packages=find_packages(),
    
    license="MIT",
    
)
