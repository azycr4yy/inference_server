from pydantic import BaseModel


class InputData(BaseModel):

    text: str
    data_id: int

class OutputData(BaseModel):
    data_id: int
    prediction: int