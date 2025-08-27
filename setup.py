from setuptools import setup, find_packages

setup(
    name="rag-llama2-system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ollama>=0.1.8",
        "requests>=2.31.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.15",
        "PyPDF2>=3.0.1",
        "rank-bm25>=0.2.2",
        "nltk>=3.8.1",
        "numpy>=1.24.3",
    ],
    author="Vishnuu A",
    author_email="A.Vishnuu@techmahindra.com",
    description="RAG system using Ollama Llama2 for PDF document processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)