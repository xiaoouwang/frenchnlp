from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='frenchnlp',
    version='0.1.5',
    description='State of the art toolchain for natural language processing in French',
    long_description_content_type="text/markdown",
    long_description=README,
    include_package_data=True,
    license='MIT',
    packages=find_packages(),
    author='Xiaoou WANG',
    author_email='xiaoouwangfrance@gmail.com',
    keywords=['text mining', 'npl', 'corpus', 'french'],
    url='https://github.com/xiaoouwang/frenchnlp',
    download_url='https://pypi.org/project/frenchnlp',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

install_requires = [
    'numpy',
    'pandas'
]


if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
