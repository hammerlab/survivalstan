from setuptools import setup, find_packages
from __future__ import print_function
from codecs import open
from os import path
import versioneer

__version__ = '0.0.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if 'git+' not in x]

readme = ""
try:
    with open(readme_path, "r") as f:
        readme = f.read()
except IOError as e:
    print(e)
    print("Failed to open %s" % readme_path)

try:
    import pypandoc
    readme = pypandoc.convert(readme, to="rst", format="md")
except ImportError as e:
    print(e)
    print("Failed to convert %s to reStructuredText", readme_filename)
    pass

setup(
    name='stanmodels',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Library of Stan Models for Computational Biology',
    long_description=long_description,
    url='https://github.com/jburos/stanmodels',
    download_url='https://github.com/jburos/stanmodels/tarball/' + __version__,
    license="http://www.apache.org/licenses/LICENSE-2.0.html",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='Jacki Novik',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='jackinovik@gmail.com'
)
