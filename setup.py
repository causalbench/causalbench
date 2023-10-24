"""
Copyright 2023 GlaxoSmithKline plc - Mathieu Chevalley, Yusuf Roohani, Patrick Schwab, Arash Mehrjou; 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name='causalbench',
    version='1.1.1',
    python_requires=">=3.8",
    packages=find_packages(),
    package_data={
        "": ["*.txt"],
        "": ["*.csv"],
        "causalscbench.data_access": ["data/*.csv"],
    },
    author='see README.txt',
    url="https://www.gsk.ai/causalbenchchallenge",
    author_email='biomedical-ai-external@gsk.com',
    license="Apache-2.0",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'causalbench_run=causalscbench.apps.main_app:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)