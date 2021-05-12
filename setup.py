from setuptools import find_packages, setup

setup(
    name="cvpr-project",
    packages=find_packages(include=["cvpr"]),
    version="0.1.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
    author="Me",
    license="MIT",
)
