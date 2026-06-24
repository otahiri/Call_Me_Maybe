"""function model"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, Union


class Func(BaseModel):
    """func class

    Attributes:
        name: name of the function
        description: description about the function
        prompt: prompt that will was provided to the model
        parameters: the parameters dictionary
    """
    name: str = Field(min_length=1)
    description: str = ""
    prompt: str = Field(min_length=1)
    parameters: Dict[str, Union[int, float, str]] = dict()

    @model_validator(mode='after')
    def validate_func(self) -> "Func":
        for p in self.parameters.values():
            if p not in ["string", "number", "integer"]:
                raise ValueError("invalid type detected")
        return self
