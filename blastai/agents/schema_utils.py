"""
Utilities for converting between JSON Schema and Pydantic models.

Provides helpers for:
- Converting JSON schemas to recursive Pydantic models with full nesting
- Generating Python type hints from JSON schema types
- String case conversions (snake_case <-> PascalCase)
- Dynamic Pydantic model creation from JSON schemas
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, create_model


def snake_to_pascal(snake_str: str) -> str:
    """
    Convert snake_case to PascalCase.
    
    Example:
        >>> snake_to_pascal("get_restaurant_details")
        'GetRestaurantDetails'
    """
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def json_type_to_python(json_type: str) -> str:
    """
    Convert JSON schema type to Python type hint string.
    
    Args:
        json_type: JSON schema type (string, number, integer, boolean, array, object)
    
    Returns:
        Python type hint as string
        
    Example:
        >>> json_type_to_python("string")
        'str'
        >>> json_type_to_python("array")
        'List[Any]'
    """
    type_map = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "array": "List[Any]",
        "object": "Dict[str, Any]"
    }
    return type_map.get(json_type, "Any")


def generate_nested_pydantic_classes(
    schema: Dict[str, Any], 
    class_prefix: str, 
    lines: List[str]
) -> str:
    """
    Recursively generate nested Pydantic classes from a JSON schema.
    Returns the type hint for the schema.
    
    This traverses the schema depth-first and generates all nested classes before
    the parent class that references them. This allows for fully type-safe nested
    structures like:
    
        class MenuItem(BaseModel):
            name: str
            price: str
        
        class RestaurantItem(BaseModel):
            name: str
            menu_items: List[MenuItem]
        
        class GetRestaurantDetailsOutput(BaseModel):
            items: List[RestaurantItem]
    
    Args:
        schema: JSON schema (can be object, array, or primitive type)
        class_prefix: Prefix for generated class names
        lines: List to append generated class definitions to
    
    Returns:
        Type hint string (e.g., "str", "List[MenuItem]", "Optional[Restaurant]")
        
    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "items": {
        ...             "type": "array",
        ...             "items": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "name": {"type": "string"},
        ...                     "price": {"type": "number"}
        ...                 },
        ...                 "required": ["name"]
        ...             }
        ...         }
        ...     },
        ...     "required": ["items"]
        ... }
        >>> lines = []
        >>> type_hint = generate_nested_pydantic_classes(schema, "Output", lines)
        >>> print("\\n".join(lines))
        class OutputItemsItem(BaseModel):
            name: str = Field(..., description="")
            price: Optional[float] = None
        <BLANKLINE>
        class Output(BaseModel):
            items: List[OutputItemsItem] = Field(..., description="")
        <BLANKLINE>
    """
    schema_type = schema.get("type")
    
    # Handle type unions (e.g., ["string", "null"])
    if isinstance(schema_type, list):
        non_null_types = [t for t in schema_type if t != "null"]
        is_nullable = "null" in schema_type
        
        if len(non_null_types) == 1:
            # Recursively handle the single non-null type
            temp_schema = dict(schema)
            temp_schema["type"] = non_null_types[0]
            inner_type = generate_nested_pydantic_classes(temp_schema, class_prefix, lines)
            return f"Optional[{inner_type}]" if is_nullable else inner_type
        else:
            # Multiple non-null types - generate Union
            type_hints = []
            for t in non_null_types:
                temp_schema = dict(schema)
                temp_schema["type"] = t
                type_hints.append(generate_nested_pydantic_classes(temp_schema, class_prefix, lines))
            union_str = f"Union[{', '.join(type_hints)}]"
            return f"Optional[{union_str}]" if is_nullable else union_str
    
    # Handle arrays
    if schema_type == "array":
        items_schema = schema.get("items", {})
        if not items_schema:
            return "List[Any]"
        
        # Recursively generate type for array items
        item_type = generate_nested_pydantic_classes(items_schema, f"{class_prefix}Item", lines)
        return f"List[{item_type}]"
    
    # Handle objects - generate a new Pydantic class
    if schema_type == "object":
        properties = schema.get("properties", {})
        if not properties:
            return "Dict[str, Any]"
        
        # First, recursively generate nested classes for all properties
        # This ensures nested classes are defined before the parent class
        prop_type_hints = {}
        for prop_name, prop_schema in properties.items():
            prop_type_hint = generate_nested_pydantic_classes(
                prop_schema, 
                f"{class_prefix}{snake_to_pascal(prop_name)}", 
                lines
            )
            prop_type_hints[prop_name] = prop_type_hint
        
        # Now generate the class definition for this object
        lines.append(f"class {class_prefix}(BaseModel):")
        
        required = schema.get("required", [])
        has_fields = False
        
        for prop_name, prop_schema in properties.items():
            has_fields = True
            prop_desc = prop_schema.get("description", "")
            is_required = prop_name in required
            prop_type_hint = prop_type_hints[prop_name]
            
            # Generate field definition
            if is_required:
                lines.append(f'    {prop_name}: {prop_type_hint} = Field(..., description="{prop_desc}")')
            else:
                if "Optional" in prop_type_hint or prop_type_hint.startswith("Union"):
                    lines.append(f'    {prop_name}: {prop_type_hint} = None')
                else:
                    lines.append(f'    {prop_name}: Optional[{prop_type_hint}] = None')
        
        if not has_fields:
            lines.append("    pass")
        
        lines.append("")
        return class_prefix
    
    # Primitive types
    return json_type_to_python(schema_type) if schema_type else "Any"


def json_schema_to_pydantic(
    schema: Dict[str, Any],
    model_name: str = "DynamicModel"
) -> Optional[type[BaseModel]]:
    """
    Convert a JSON Schema to a Pydantic model.
    
    Handles:
    - Simple types (string, number, integer, boolean, array, object)
    - Nullable types (["string", "null"])
    - Required vs optional fields
    - Nested objects (basic support - browser-use handles full validation)
    
    Args:
        schema: JSON Schema dictionary with properties and required fields
        model_name: Name for the generated Pydantic model
        
    Returns:
        Pydantic model class, or None if schema is invalid/empty
        
    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string", "description": "User name"},
        ...         "age": {"type": "integer"},
        ...         "email": {"type": ["string", "null"]}
        ...     },
        ...     "required": ["name"]
        ... }
        >>> Model = json_schema_to_pydantic(schema, "User")
        >>> instance = Model(name="Alice", age=30, email=None)
    """
    if not isinstance(schema, dict):
        return None
    
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    if not properties:
        return None
    
    # Build Pydantic field definitions
    field_defs = {}
    
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        prop_desc = prop_schema.get("description", "")
        is_required = prop_name in required
        
        # Handle nullable types: ["string", "null"] or ["number", "null"]
        is_nullable = False
        if isinstance(prop_type, list):
            # Filter out "null" and get the actual type
            actual_types = [t for t in prop_type if t != "null"]
            is_nullable = "null" in prop_type
            prop_type = actual_types[0] if actual_types else "string"
        
        # Map JSON schema types to Python types
        type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        python_type = type_map.get(prop_type, str)
        
        # Handle array items schema if present (basic support)
        # Browser-use will handle full nested validation
        if prop_type == "array" and "items" in prop_schema:
            # For now, use list - browser-use will handle nested structure
            python_type = list
        
        # Build field definition with proper Optional handling
        if is_nullable or not is_required:
            # Nullable or optional field
            from typing import Optional as OptionalType
            field_defs[prop_name] = (OptionalType[python_type], Field(None, description=prop_desc))
        else:
            # Required, non-nullable field
            field_defs[prop_name] = (python_type, Field(..., description=prop_desc))
    
    # Create dynamic Pydantic model
    if not field_defs:
        return None
    
    return create_model(model_name, **field_defs)


def pydantic_to_json_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model to JSON Schema.
    
    Args:
        model: Pydantic model class
        
    Returns:
        JSON Schema dictionary
        
    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> schema = pydantic_to_json_schema(User)
        >>> schema["properties"]["name"]["type"]
        'string'
    """
    return model.model_json_schema()


__all__ = [
    "snake_to_pascal",
    "json_type_to_python",
    "generate_nested_pydantic_classes",
    "json_schema_to_pydantic",
    "pydantic_to_json_schema"
]
