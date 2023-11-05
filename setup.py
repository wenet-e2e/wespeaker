from setuptools import setup, find_packages

requirements = [
    "tqdm",
    "onnxruntime>=1.12.0",
    "python-speech-features>=0.6",
    "scipy>=1.5.2",
]

setup(
    name="wespeaker",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wespeaker = wespeaker.cli.speaker:main",
    ]},
)
