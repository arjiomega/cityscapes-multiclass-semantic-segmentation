import torch
import pytest

import src.model.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dim_expectations() -> tuple[tuple, tuple]:
    BATCH_SIZE = 1
    INPUT_CHANNELS = 3
    HEIGHT, WIDTH = 112, 112

    OUTPUT_CHANNELS = 2

    INPUT_SHAPE = (BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    OUTPUT_SHAPE = (BATCH_SIZE, OUTPUT_CHANNELS, HEIGHT, WIDTH)

    return INPUT_SHAPE, OUTPUT_SHAPE


@pytest.fixture(params=models._list_of_models)
def model(request):
    """Iteratively load models in models module.
    Ex. [Model_A, Model_B, ...]
    """
    return request.param


def test_models_io_shape(dim_expectations, model):
    """Test all the models in models module iteratively."""

    INPUT_SHAPE, OUTPUT_SHAPE = dim_expectations

    # (batch size, channels (rgb), height, width)
    x = torch.randn(INPUT_SHAPE, device=device)

    load_model = model(in_channels=3, out_channels=2).to(device)
    predict = load_model(x)

    failed_test_output = f"model: {model} failed io shape test."
    assert predict.shape == OUTPUT_SHAPE, failed_test_output


def test_models_output_not_softmax(dim_expectations, model):
    """Test that the output tensor of the model is not yet passed to softmax.
    Logits is apparently better for calculating loss functions numerically.
    This is not as important but it is better to make sure all models pass
    the same expectations.
    """

    INPUT_SHAPE, OUTPUT_SHAPE = dim_expectations

    # (batch size, channels (rgb), height, width)
    x = torch.randn(INPUT_SHAPE, device=device)

    load_model = model(in_channels=3, out_channels=2).to(device)

    # (channels (rgb), height, width)
    predict = load_model(x).squeeze(0)

    # (height, width)
    class_sum = torch.sum(predict, dim=0)

    ones = torch.ones(OUTPUT_SHAPE[-2], OUTPUT_SHAPE[-1]).to(device)

    equality_check = torch.eq(class_sum, ones)

    # class_sum should not equal to one since we are trying to prevent models
    # that include softmax
    assert not torch.all(equality_check)
