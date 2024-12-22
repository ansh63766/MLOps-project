from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line for line in f.read().splitlines() if line != '-e .']

setup(
    name='mlops_project',
    version='0.1.0',
    description='Its simple mlops project',
    author='Shivansh Gupta',
    author_email='shivanshguptas285@gmail.com',
    url='https://github.com/ansh63766/MLOps-project/',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
