from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='data_science_pro',
    version='0.1.0',
    description='LLM-powered modular data science pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'langchain',
        'openai',
        'python-dotenv',
        'joblib',
        'imbalanced-learn'
    ],
    entry_points={
        'console_scripts': [
            'data-science-pro=pipeline:main'
        ]
    },
    python_requires='>=3.8',
)
