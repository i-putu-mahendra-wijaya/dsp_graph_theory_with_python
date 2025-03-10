from setuptools import setup, find_packages

setup(
    name="dsp_intro_to_graph",
    version="0.0.0",
    author="I Putu Mahendra Wijaya",
    author_email="i.putu.mahendra@gmail.com",
    description="dsp_intro_to_graph",
    url="",
    packages=find_packages(
        include=[
            "dsp_intro_to_graph",
            "dsp_intro_to_graph.*"
        ]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.10.5",
)