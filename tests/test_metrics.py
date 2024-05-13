import torch
import pytest
from src.model.metrics import iou, Metric, Metrics

DEVICE = "cpu"


@pytest.fixture
def sample_inputs():
    NUM_CLASSES = 3
    DATA_DIM = 10
    BATCH_SIZE = 4
    SEED = 0

    torch.manual_seed(SEED)

    sample_classmap = {0: "class 1", 1: "class 2", 2: "class 3"}

    size = (BATCH_SIZE, NUM_CLASSES, DATA_DIM, DATA_DIM)

    test_true_mask = torch.randint(0, 2, size=size, device=DEVICE)
    test_predict_mask = torch.randn(*size, device=DEVICE)

    return test_true_mask, test_predict_mask, sample_classmap


def test_iou(sample_inputs):

    test_true_mask, test_predict_mask, sample_classmap = sample_inputs

    output = iou(test_true_mask, test_predict_mask, sample_classmap, DEVICE)

    EXPECTATION = {
        "IoU": {
            "class 1": 0.2509225010871887,
            "class 2": 0.24902723729610443,
            "class 3": 0.23636363446712494,
        },
        "mIoU": 0.24543779095013937,
    }

    assert output == EXPECTATION, "Expected output from iou was not met."


def test_metrics_solver(sample_inputs):

    test_true_mask, test_predict_mask, sample_classmap = sample_inputs

    metric1 = Metric(name="iou", metric_fn=iou)
    metrics_solver = Metrics(sample_classmap, DEVICE, metric1)

    EPOCHS = 3

    for epoch in range(EPOCHS):
        metrics_solver.calculate_metrics(test_true_mask, test_predict_mask)

    metrics = metrics_solver.get_metrics()

    EXPECTATIONS = {
        "iou": {
            "byclass": {
                "class 1": 0.2509225010871887,
                "class 2": 0.24902723729610443,
                "class 3": 0.23636363446712494,
            },
            "mean": 0.24543779095013937,
        }
    }

    assert (
        metrics == EXPECTATIONS
    ), "Wrong metric(s) from metrics solver. See classes Metric and Metrics."
