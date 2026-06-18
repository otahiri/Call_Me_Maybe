from pydantic import BaseModel, Field
from typing import Literal, Dict


class Param(BaseModel):
    param_type: Literal["string", "number", "integer"]


class Func(BaseModel):
    name: str = Field(min_length=1)
    description: str
    params: Dict[str, Param]
