import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eighttrack",
    version="0.0.1",
    author="JC Jimenez",
    author_email="jc.jimenez@microsoft.com",
    description="A simple package to boostrap an object detection and tracking pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcjimenez/8track",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'rtree'
    ],
    tests_require=[
        'mock'
    ]
)
