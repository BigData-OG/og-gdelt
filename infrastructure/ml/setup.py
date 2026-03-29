from setuptools import find_packages
from setuptools import setup

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    package_data={
        'trainer': ['*.json']
    },
    include_package_data=True,
    description='GDELT Vertex AI training application.'
)
