from pydantic import BaseModel
from pydantic import ConfigDict


class StrictPydanticBaseModel(BaseModel):
    """Pydantic base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")
