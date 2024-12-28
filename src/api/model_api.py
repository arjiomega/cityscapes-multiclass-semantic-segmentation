import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request

from src.api import api_utils
from src.model.inference import Inference

logger = logging.getLogger("uvicorn.error")

api_dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to initialize and clean up resources for the FastAPI application.

    This function initializes the Inference object and stores it in a global dictionary.
    It yields control back to the FastAPI application and clears the resources upon application shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.
    """

    inference = Inference.from_local_model("08152024")
    api_dict["inference"] = inference
    yield
    api_dict.clear()


# Initialize API Server
app = FastAPI(
    title="ML Model",
    description="Description of the ML Model",
    version="0.0.1",
    lifespan=lifespan,
)


@app.post("/predict")
async def predict(request: Request):
    """
    Endpoint to perform model inference on the provided input.

    This endpoint receives a request containing input data in hex format and metadata,
    processes the data according to the specified format, and returns the model's prediction.

    Args:
        request (Request): The FastAPI request object containing the input data.

    Returns:
        dict: A dictionary containing the prediction result in hex format.

    Notes:
        Hex was chosen as an input format instead of torch.Tensor.tolist() due to performance considerations.
    """

    response = await request.json()
    input_ = response["input"]  # str.encode(response["input"])
    shape_ = response["shape"]
    dtype = response["dtype"]
    request_format = response["request_format"]

    # Update hex input from request
    match request_format:
        case "tensor":
            logger.info("converting hex input to torch tensor...")
            processed_data = api_utils.hex2tensor(input_, shape_, dtype)
        case "numpy":
            logger.info("converting hex input to numpy array...")
            processed_data = api_utils.hex2numpy(input_, shape_, dtype)
        case _:
            logger.debug(
                f"""
                Requested format '{request_format}' invalid. 
                Available formats are 'tensor' and 'numpy'
                """
            )

            return False

    # # get a prediction
    logger.info("predicting from input...")
    prediction = api_dict["inference"].predict(processed_data)
    logger.debug(f"prediction shape: {prediction.shape}")
    logger.debug(f"prediction dtype: {prediction.dtype}")
    logger.debug(f"{torch.unique(prediction)=}")

    logger.debug(f"processed data shape: {processed_data.shape}")
    test = api_utils.tensor2hex(prediction)

    return {"result": test}
