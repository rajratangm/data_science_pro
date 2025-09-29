from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='data_science_pro',
    version='0.2.0',
    description='LLM-powered modular data science pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Rajratan More',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'imbalanced-learn',
        'joblib',
        'langchain',
        'langchain-community',
        'langchain-openai',
        'langgraph',
        'chromadb',
        'openai',
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'data-science-pro=data_science_pro.pipeline:main'
        ]
    },
    python_requires='>=3.8',
)
