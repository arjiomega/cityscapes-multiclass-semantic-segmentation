import platform
from setuptools import find_packages, setup

# solution from https://github.com/pytorch/pytorch/issues/50718
def install_torch(package: str, version: str = ''):
    """This is needed to make setup compatible with M1 chip."""
    cuda = "arm" not in platform.platform()
    python_version = ''.join(platform.python_version().split('.')[:2])

    return ''.join([
        f'{package} @ https://download.pytorch.org/whl/',
        f'cu118/' if cuda else '',
        f'{package}',
        f'-{version}' if version else '',
        '%2Bcu118' if cuda else '',
        f'-cp{python_version}-cp{python_version}',
        'm' if int(python_version) <= 37 else '',
        '-linux_x86_64.whl',
    ])

setup(
    name="src",
    python_requires=">3.11.0",
    packages=find_packages(),
    version="1.0.0",
    install_requires = [
        "matplotlib==3.8.4",
        "scikit-image==0.23.2",
        "opencv-python==4.9.0.80",
        install_torch('torch', '2.3.0'),
        install_torch('torchvision', '0.18.0'),
        # "torch[cu118]==2.3.0 @ https://download.pytorch.org/whl/cu118#egg=repo",
        # "torchvision[cu118]==0.18.0 @ https://download.pytorch.org/whl/cu118",
        "torchmetrics==1.4.0",
        "torchinfo==1.8.0",
        "albumentations==1.4.6",
        "pydantic==2.7.1",
        "tqdm==4.66.4",
        "mlflow==2.12.1", 
    ],
    extras_require={
        "dev": [
            "typeguard==4.2.1",
            "pytest==8.2.0",
            "black==24.4.2",
            "pre-commit==3.7.0",
            "notebook==7.1.3"
        ]
    },
    # dependency_links=[
    #     "https://download.pytorch.org/whl/cu118"
    # ]
)
