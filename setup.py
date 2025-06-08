from setuptools import setup, find_packages


ascii_logo = """
████████╗████████╗███████╗███████╗███╗   ███╗
╚══██╔══╝╚══██╔══╝██╔════╝██╔════╝████╗ ████║ 
   ██║      ██║   █████╗  █████╗  ██╔████╔██║ 
   ██║      ██║   ██╔══╝  ██╔══╝  ██║╚██╔╝██║ 
   ██║      ██║   ██║     ███████╗██║ ╚═╝ ██║ 
   ╚═╝      ╚═╝   ╚═╝     ╚══════╝╚═╝     ╚═╝ 
"""

print(ascii_logo)

requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
    name="ttfemesh",
    version="0.1.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10,<3.13",
    author="Mazen Ali",
    author_email="mazen.ali90@gmail.com",
    description="A Python library for generating tensor train representations of finite element meshes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MazenAli/TT-FEMesh",
    project_urls={
        "Source": "https://github.com/MazenAli/TT-FEMesh",
        "Documentation": "https://tt-femesh.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="tensor train finite elements meshing simulation",
)
