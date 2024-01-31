from setuptools import setup, find_packages

setup(
    
    name = "whalegrad",
    version = "0.0.1",
    author = "Saurabh Aone",
    packages = find_packages(),
    license = "MIT",
    description = "A deep learning framework from scratch",
    long desxription = open("README.md").read(),
    install_requires = ["numpy"],
    
    )