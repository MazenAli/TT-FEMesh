from setuptools import setup, find_packages

# ASCII logo
ascii_logo = """
████████╗███╗   ██╗███████╗███████╗███╗   ███╗
╚══██╔══╝████╗  ██║██╔════╝██╔════╝████╗ ████║ 
   ██║   ██╔██╗ ██║█████╗  █████╗  ██╔████╔██║ 
   ██║   ██║╚██╗██║██╔══╝  ██╔══╝  ██║╚██╔╝██║ 
   ██║   ██║ ╚████║██║     ███████╗██║ ╚═╝ ██║ 
   ╚═╝   ╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝     ╚═╝ 
"""

print(ascii_logo)

# Setup configuration
setup(
    name="tnfemesh",
    version="0.1.0",
    description="A Python library for tensor network-based finite element meshing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mazen Ali",
    author_email="mazen.ali90@gmail.com",
    url="https://github.com/MazenAli/tnfemesh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
    ],
    python_requires=">=3.9, <3.11",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    project_urls={
        "Source": "https://github.com/MazenAli/tnfemesh",
        "Documentation": "https://github.com/MazenAli/tnfemesh",
    },
    keywords="tensor networks finite elements meshing simulation",
)
