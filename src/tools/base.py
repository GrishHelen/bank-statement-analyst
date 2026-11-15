"""Контракт инструмента и выполнение вызова с проверкой аргументов.

Модель видит только имя, описание и JSON-схему аргументов. Код инструмента
модель не видит: аргументы проверяются здесь, до выполнения функции.
Любой исход возвращается модели как данные в одном формате:

    {"ok": true,  "tool": "...", "result": {...}}
    {"ok": false, "tool": "...", "error": {"code": "...", "message": "...", "details": [...]}}
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, ValidationError

ToolKind = Literal["read", "compute", "action"]


class ToolError(Exception):
    """Ожидаемый отказ инструмента: понятная причина вместо падения агента."""

    def __init__(self, code: str, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    kind: ToolKind
    args_model: type[BaseModel]
    function: Callable[[Any], dict[str, Any]]
    examples: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def schema(self) -> dict[str, Any]:
        return simplify_schema(self.args_model.model_json_schema())


@dataclass(frozen=True)
class ToolTrace:
    """Запись о выполненном шаге для вывода пользователю и отладки."""

    name: str
    arguments: Any
    ok: bool
    summary: str


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def specifications(self) -> list[dict[str, Any]]:
        """Описания инструментов в нейтральном формате: имя, описание, схема, примеры."""
        result: list[dict[str, Any]] = []
        for tool in self.tools.values():
            result.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema(),
                "examples": [{"request": request, "params": params} for request, params in tool.examples],
            })
        return result

    def execute(self, name: str, raw_arguments: Any) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return error_result(name, "unknown_tool", f"Инструмента {name} нет.",
                                ["Доступны: " + ", ".join(self.tools)])
        arguments = raw_arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                return error_result(name, "invalid_arguments", "Аргументы не являются корректным JSON.")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return error_result(name, "invalid_arguments", "Аргументы должны быть объектом JSON.")
        try:
            parsed = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            return error_result(name, "invalid_arguments", "Аргументы не прошли проверку.",
                                explain_validation_error(exc))
        try:
            result = tool.function(parsed)
        except ToolError as exc:
            return error_result(name, exc.code, exc.message, exc.details)
        except Exception:  # noqa: BLE001 — сбой инструмента не должен ронять агента
            return error_result(name, "tool_failed", "Инструмент завершился с внутренней ошибкой.")
        return {"ok": True, "tool": name, "result": result}


def error_result(name: str, code: str, message: str, details: list[str] | None = None) -> dict[str, Any]:
    return {"ok": False, "tool": name, "error": {"code": code, "message": message, "details": details or []}}


def explain_validation_error(exc: ValidationError) -> list[str]:
    """Перевести ошибки pydantic в короткие сообщения на русском."""
    messages: list[str] = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item["loc"]) or "аргументы"
        kind = item["type"]
        context = item.get("ctx") or {}
        if kind == "missing":
            text = "обязательный аргумент"
        elif kind == "extra_forbidden":
            text = "неизвестный аргумент"
        elif kind.startswith("date"):
            text = "ожидается существующая дата в формате ГГГГ-ММ-ДД"
        elif kind == "literal_error":
            text = f"допустимые значения: {context.get('expected', '')}"
        elif kind in ("int_parsing", "int_from_float", "int_type"):
            text = "ожидается целое число"
        elif kind in ("float_parsing", "float_type", "decimal_parsing"):
            text = "ожидается число"
        elif kind == "greater_than_equal":
            text = f"должно быть не меньше {context.get('ge')}"
        elif kind == "less_than_equal":
            text = f"должно быть не больше {context.get('le')}"
        elif kind == "string_too_long":
            text = f"не длиннее {context.get('max_length')} символов"
        elif kind == "value_error":
            text = str(context.get("error", item["msg"]))
        else:
            text = item["msg"]
        messages.append(f"{location}: {text}")
    return messages


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Схема без anyOf/null и служебных title: так её надёжнее понимают разные API."""
    properties: dict[str, Any] = {}
    for name, prop in schema.get("properties", {}).items():
        prop = dict(prop)
        variants = prop.pop("anyOf", None)
        if variants:
            for variant in variants:
                if variant.get("type") != "null":
                    prop = {**variant, **prop}
                    break
        prop.pop("title", None)
        if prop.get("default", "missing") is None:
            prop.pop("default")
        properties[name] = prop
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if schema.get("required"):
        result["required"] = list(schema["required"])
    return result
