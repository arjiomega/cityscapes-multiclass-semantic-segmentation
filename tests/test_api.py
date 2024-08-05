import torch

from src.api import api_utils


def test_hex_tensor_vice_versa_conversion():
    sample_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    shape = list(sample_input.shape)
    dtype = str(sample_input.dtype)

    # convert to hex
    to_hex = api_utils.tensor2hex(sample_input)

    # convert back to tensor
    to_tensor = api_utils.hex2tensor(to_hex, shape, dtype)

    assert torch.equal(sample_input, to_tensor)
