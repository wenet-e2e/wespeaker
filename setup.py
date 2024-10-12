from setuptools import setup, find_packages

requirements = [
    "tqdm",
    "kaldiio",
    "hdbscan==0.8.37",
    "umap-learn==0.5.6",
    "torch>=1.12.0",
    "torchaudio>=0.12.0",
    "silero-vad",
]

setup(
    name="wespeaker",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wespeaker = wespeaker.cli.speaker:main",
        ]
    },
)
