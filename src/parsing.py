from pydantic import BaseModel, model_validator


class functions(BaseModel):
    functions: list[dict]

    @model_validator(mode="after")
    def validate_funcs(self):
        valid_keys = ["name", "description", "parameters", "returns"]
        valid_types = ["string", "number", "integer"]
        for f in self.functions:
            if not len(f.keys()) == len(valid_keys):
                raise ValueError
            elif any(item not in valid_keys for item in f.keys()):
                raise ValueError
            params = f["parameters"]
            for value in params.values():
                try:
                    param_type = value["type"]
                except KeyError:
                    raise ValueError("parameters should have type")
                if param_type not in valid_types:
                    raise ValueError("invalid type detected")
            try:
                return_type = f["returns"]["type"]
                print(return_type)
            except KeyError:
                raise ValueError("return field missing")
            if return_type not in valid_types:
                raise ValueError("return type is not valid")

        return self
