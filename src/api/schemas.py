from pydantic import BaseModel


class InputData(BaseModel):
    text: str

