from setuptools import setup, find_packages #type: ignore

setup(
    name="mrplot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MRI visualization toolkit for generating slice plots and videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mri-plot",
    packages=find_packages(),
    package_data={
        "mrplot": [
            "templates/*",
            "templates/*.html",
            "configs/*.json"
        ]
    },
    include_package_data=True,
    install_requires=[
        "nibabel>=4.0",
        "numpy>=1.21",
        "matplotlib>=3.5",
        "imageio>=2.19",
        "imageio-ffmpeg>=0.4.7",
        "pytest>=7.0"  # For testing
    ],
    entry_points={
        "console_scripts": [
            "mrplot = mrplot.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 