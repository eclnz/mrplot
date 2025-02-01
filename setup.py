from setuptools import setup, find_packages

setup(
    name="mri_plot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MRI visualization toolkit for generating slice plots and videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mri-plot",
    packages=find_packages(),
    install_requires=[
        "nibabel>=4.0",
        "numpy>=1.21",
        "matplotlib>=3.5",
        "imageio>=2.19",
        "pytest>=7.0"  # For testing
    ],
    entry_points={
        "console_scripts": [
            "mriplot = mri_plot.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 