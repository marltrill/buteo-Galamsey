import setuptools


def readme():
    try:
        with open("README.md") as f:
            return f.read()
    except IOError:
        return ""


setuptools.setup(
    name="Buteo",
    version="0.0.1",
    author="Casper Fibaek",
    author_email="casperfibaek@gmail.com",
    description="An Earth Observation toolbox for Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/casperfibaek/buteo",
    project_urls={
        "Bug Tracker": "https://github.com/casperfibaek/buteo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Alpha",
    ],
    packages=["buteo"],
    install_requires=[
        "numpy",
        "numba",
        "pandas",
        "sentinelsat",
        "tqdm",
    ],
    include_package_data=True,
)

# python -m build
# python -m twine upload --repository testpypi dist/*
# pdoc3 --html --output-dir docs --config show_source_code=False buteo