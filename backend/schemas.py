
from pydantic import BaseModel, Field, model_validator
from typing import Literal


class Constraint(BaseModel):
    name: str
    coefficients: dict[str, float]
    sense: Literal["==", "<=", ">=", "<", ">"]
    rhs: float


class OptimizationModel(BaseModel):
    title: str

    variables: list[str] = Field(min_length=1, max_length=20)
    objective: dict[str, float]

    sense: Literal["min", "max"] = "min"
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_variables(self):
        names = set(self.variables)

        # Check duplicate variable names
        if len(names) != len(self.variables):
            raise ValueError("Duplicate variable names")

        # Check objective variables
        if not set(self.objective).issubset(names):
            raise ValueError("Unknown objective variable")

        # Check constraint variables
        for constraint in self.constraints:
            if not set(constraint.coefficients).issubset(names):
                raise ValueError(
                    f"Unknown variable in constraint: {constraint.name}"
                )

        return self
