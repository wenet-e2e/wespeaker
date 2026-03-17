from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="wespeaker",
    version="0.0.1",
    description=("WeSpeaker: A Research and Production Oriented "
                 "Speaker Embedding Learning Toolkit"),
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.8",
    author="WeSpeaker Contributors",
    keywords=["speaker", "embedding", "verification", "diarization", "speech"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "tqdm",
        "kaldiio",
        "hdbscan>=0.8.40",
        "umap-learn==0.5.6",
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "silero-vad",
        "s3prl",
        "openai-whisper",
        "peft",
        "accelerate",
    ],
    extras_require={
        "dev": [
            "flake8",
            "pre-commit",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/wenet-e2e/wespeaker",
        "Documentation": "http://wenet.org.cn/wespeaker",
        "Repository": "https://github.com/wenet-e2e/wespeaker",
        "Issues": "https://github.com/wenet-e2e/wespeaker/issues",
    },
    entry_points={
        "console_scripts": [
            "wespeaker=wespeaker.cli.speaker:main",
        ],
    },
    packages=find_packages(include=["wespeaker*"]),
    include_package_data=True,
)
