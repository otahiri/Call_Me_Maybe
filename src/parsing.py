from pydantic import BaseModel, model_validator


class functions(BaseModel):
    functions: list[dict]

    @model_validator(mode='after')
    def validate_funcs(self):
        valid_keys = ['name', 'description', 'parameters', 'returns']
        valid_types = []
        for f in self.functions:
            if not len(f.keys()) == len(valid_keys):
                raise ValueError
            elif  any(item not in valid_keys for item in f.keys()):
                raise ValueError
            params = f['parameters']
            for key, value in params.items():
                param_type = value['type']
                if param_type not in 

        return self


