import setuptools

setuptools.setup(
    name="whitening",
    version="0.1.3",
    url="https://github.com/remolek/whitening",

    author="Jeremi Ochab",
    author_email="jeremi.ochab@uj.edu.pl",

    description="ZCA, PCA and Cholesky whitening",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
)
