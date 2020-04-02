import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()
with open('requirements.txt', 'r') as f:
    requirements = f.read().strip('\n').split('\n')

setuptools.setup(
    name='pyPLUTOplus',
    version='2020.4.2',
    author='Gabriel Pelouze',
    author_email='gabriel.pelouze@kuleuven.be',
    description='Additions to pyPLUTO',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gpelouze/pyPLUTOplus',
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
)
