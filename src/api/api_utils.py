import numpy as np
import torch


def tensor2hex(image: torch.Tensor) -> hex:
    tensor2numpy = image.detach().numpy()
    numpy2hex = tensor2numpy.tobytes().hex()

    return numpy2hex


def hex2tensor(api_output: hex, shape: list[int], dtype: str) -> torch.Tensor:
    input_bytes = bytes.fromhex(api_output)

    dtype = torch.float32 if "float32" in dtype else torch.float64

    torch_tensor = torch.frombuffer(input_bytes, dtype=dtype).reshape(*shape)

    return torch_tensor


def hex2numpy(api_output: hex, shape: list[int], dtype: str) -> np.ndarray:
    input_bytes = bytes.fromhex(api_output)

    dtype = np.float32 if "float32" in dtype else np.float64

    torch_tensor = np.frombuffer(input_bytes, dtype=dtype).reshape(*shape)

    return torch_tensor
