from typing import Callable

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from .iou import iou
from .metrics import Metric, Metrics
