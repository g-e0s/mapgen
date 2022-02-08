from setuptools import setup, find_packages

VERSION = "1.0.0"

with open("README.md") as readme_file:
    readme = readme_file.read()

install_requires = [
    "numpy>=1.20.0",
    "matplotlib>=3.4.0",
    "gym>=0.20.0",
    "imageio-ffmpeg>=0.4.4",
    "imageio>=2.8.0"
]

setup_args = dict(
    name="mapgen",
    version=VERSION,
    description="A simple 2D map generator environment for RL algorithms",
    long_description_content_type="text/markdown",
    long_description=readme,
    install_requires=install_requires,
    python_requires=">=3.8.0",
    packages=find_packages(),
    keywords=["gym"],
    url="https://github.com/g-e0s/mapgen",
)

if __name__ == "__main__":
    setup(**setup_args)
