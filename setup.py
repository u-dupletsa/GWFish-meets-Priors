import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GWFish-meets-Priors",
    version="0.0.1",
    author="Ulyana Dupletsa",
    author_email="ulyana.dupletsa@gssi.it",
    description="Including priors on GWFish results",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/u-dupletsa/GWFish-meets-Priors",
    project_urls={
        "GSSI website": "https://www.gssi.it",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
