import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd",
    version="0.1.0",
    author="Atanu Sarkar",
    author_email="mratanusarkar@gmail.com",
    description="An experimentation on top of Andrej Karpathy's micrograd (https://github.com/karpathy/micrograd)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mratanusarkar/Learning-DeepLearning/tree/main/Neural%20Networks%20-%20Zero%20to%20Hero/micrograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)