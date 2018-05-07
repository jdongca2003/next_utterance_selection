from setuptools import setup, find_packages
import os
import sys
import shutil

# Handle the long description (read from README.txt, which is created by converting README.md)
long_description = 'answer selection'

if os.path.exists('README.txt'):
    long_description = open('README.txt').read()

setup(

    # Package structure
    #
    # find_packages searches through a set of directories 
    # looking for packages
    packages = find_packages(exclude = ['*.tests', '*.tests.*', 'tests.*', 'tests']),
    
    # package_dir directive maps package names to directories.
    # package_name:package_directory
    entry_points = {
         'console_scripts':[]
    },

    package_data={},
 
    # Not all packages are capable of running in compressed form, 
    # because they may expect to be able to access either source 
    # code or data files as normal operating system files.
    zip_safe = False,
    
    install_requires = ['tensorflow-gpu==1.4.0'],

    # Tests
    #
    # Tests must be wrapped in a unittest test suite by either a
    # function, a TestCase class or method, or a module or package
    # containing TestCase classes. If the named suite is a package,
    # any submodules and subpackages are recursively added to the
    # overall test suite.

    #test_suite = 'answer_selection.tests',

    # Download dependencies in the current directory
    #tests_require = [],

    name = "answer_selection",
    version = "0.0.1",

    # metadata for upload to PyPI
    author = "Jianxiong Dong",
    author_email = "jdongca2003@@gmail.com",
    #description = "",
    #license = "",
    #keywords = "",
    #url = "",   # project home page, if any
    #long_description = long_description
    # could also include download_url, classifiers, etc.
)
