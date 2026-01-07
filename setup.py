from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirement_list:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print(f"Error: requirements.txt file not found.")
    
    return requirement_list

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="CHARAN",
    author_email="charancherry.s703@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)

