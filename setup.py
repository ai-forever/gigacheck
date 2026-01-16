from pathlib import Path

import pkg_resources as pkg
from setuptools import find_packages, setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [f"{x.name}{x.specifier}" for x in pkg.parse_requirements((PARENT / "requirements.txt").read_text())]


exec(open("gigacheck/version.py").read())
setup(
    name="gigacheck",  # name of pypi package
    version=__version__,  # version of pypi package # noqa: F821 # type: ignore
    python_requires=">=3.11",
    description="",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    project_urls={},
    author="Layer Team, SberAI",
    author_email="",
    packages=find_packages(include=["gigacheck", "gigacheck.*"]),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Attribution-ShareAlike 4.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="machine-learning, deep-learning, ML, DL, AI, transformer",
)
