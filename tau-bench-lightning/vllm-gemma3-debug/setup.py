from setuptools import setup, find_packages

setup(
    name="vllm-gemma3-debug",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project to debug the gemma3 agent model using the vllm provider.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "litellm",
        "vllm",
        # Add any other dependencies required for your project here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)