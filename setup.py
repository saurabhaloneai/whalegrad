from setuptools import setup, find_packages

setup(
    
    name = "whalegrad",
    version = "0.0.1",
    author = "Saurabh Aone",
    packages = find_packages(),
    license = "MIT",
    description = "A deep learning framework from scratch",
    long_description=open("README.md").read(),
    long_description_content_type = "text/x-rst",
    install_requires = ["numpy"],
    
    )