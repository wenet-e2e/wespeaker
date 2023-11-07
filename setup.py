from setuptools import setup, find_packages

requirements = [
    "tqdm",
    "onnxruntime>=1.12.0",
    "librosa>=0.8.0",
    "silero-vad @ git+https://github.com/pengzhendong/silero-vad.git",
]

setup(
    name="wespeaker",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wespeaker = wespeaker.cli.speaker:main",
    ]},
)
