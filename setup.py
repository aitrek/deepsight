import pathlib
from setuptools import setup, find_packages


def get_requires():
    requires = []
    with open("requirements.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requires.append(line)
    return requires


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="deepsight",
    version="0.0.1",
    description="Integrated Object Detectors",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/aitrek/deepsight",
    author="aitrek",
    author_email="aitrek.zh@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    # install_requires=get_requires()
)
