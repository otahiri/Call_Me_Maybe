from pydantic import BaseModel, Field, model_validator
from typing import Dict, Union


class Func(BaseModel):
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
