from dataclasses import asdict, fields, is_dataclass
from typing import Type, TypeVar


T = TypeVar("T")


def dataclass_to_dict(instance) -> dict:
    if not is_dataclass(instance):
        raise TypeError("dataclass_to_dict expects a dataclass instance")
    return asdict(instance)


def dataclass_from_dict(model_cls: Type[T], data: dict) -> T:
    field_names = {field.name for field in fields(model_cls)}
    filtered = {key: value for key, value in data.items() if key in field_names}
    return model_cls(**filtered)
