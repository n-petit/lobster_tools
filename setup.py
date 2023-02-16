#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Nicolas Petit",
    author_email='nicolas.petit@keble.ox.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python tools for working with Limit Order Book data from LOBSTER.",
    entry_points={
        'console_scripts': [
            'lobster_tools=lobster_tools.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lobster_tools',
    name='lobster_tools',
    packages=find_packages(include=['lobster_tools', 'lobster_tools.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/n-petit/lobster_tools',
    version='1.0',
    zip_safe=False,
)
