def parse_return_line(return_line: str):
    return_line = return_line.replace("Returns:", "").strip()
    parts = return_line.split(":", 1)
    if len(parts) > 1:
        return_type, return_description = parts[0].strip(), parts[1].strip()
        return_type = return_type.lstrip("(").rstrip(")")
    else:
        return_description = ""
        return_type = return_line

    return_type, _ = translate_data_type(return_type)
    payload = {"description": return_description}
    if isinstance(return_type, str):
        payload["type"] = return_type
    else:
        payload["oneOf"] = return_type

    return payload


def parse_param_line(param_line: str):
    enum = None
    parts = param_line.split(":", 1)
    if len(parts) > 1:
        param_name, param_description = parts[0].strip(), parts[1].strip()
        name_and_type = param_name.split(" ", 1)
        if len(name_and_type) > 1:
            param_name = name_and_type[0]
            if param_name == "key_name":
                enum = ["ALLOWED_VALUES_FOR_KEY_NAME"]
                param_type = "string"
            else:
                param_type = name_and_type[1].lstrip("(").rstrip(")")
                param_type, enum2 = translate_data_type(param_type)
                if enum2 is not None:
                    enum = enum2
        else:
            param_type = "object"
        payload = {"description": param_description}
        if isinstance(param_type, str):
            payload["type"] = param_type
        else:
            payload["oneOf"] = param_type
    else:
        raise Exception(f"Problem with docstring line: {param_line}")
    if enum is not None:
        payload["enum"] = enum
    return param_name, payload


def translate_data_type(data_type: str):
    enum = None
    data_type = data_type.replace("<class", "").replace(">", "").strip()
    data_type = data_type.replace("'", "").replace('"', "")
    if data_type.startswith("Union"):
        data_type = data_type.replace("Union", "").replace("[", "").replace("]", "")
        subtypes = data_type.split(", ")
        data_type = [
            {"type": open_api_conversion_dict.get(s, "object")} for s in subtypes
        ]
    elif data_type.startswith("typing.Literal"):
        enum = data_type.replace("typing.Literal[", "").replace("]", "").split(", ")
        data_type = "string"
    else:
        data_type = open_api_conversion_dict.get(data_type, "object")
    return data_type, enum


def parse_google_style_docstring(docstring: str) -> dict:
    """
    Parse a Google-style docstring to extract the description, parameters, and return type.
    Args:
        docstring (str): Google-style docstring.

    Returns:
        dict: A dictionary containing parsed data (description, parameters, return type).
    """
    parsed_info = {
        "description": "",
        "parameters": {},
        "output_parameter": "",
    }

    if docstring is None:
        return parsed_info

    lines = docstring.strip().split("\n")
    description_lines = []
    params_lines = []
    returns_line = ""

    state = "description"  # initial state to capture description

    for line in lines:
        if line.strip().startswith("Args:"):
            state = "params"
        elif line.strip().startswith("Returns:"):
            state = "returns"
        elif line.strip().startswith("Raises:"):
            state = "raises"

        if state == "description":
            description_lines.append(line.strip())
        elif state == "params":
            if line.strip() == "Args:":
                continue
            if line.strip().startswith("-"):
                params_lines[
                    -1
                ] += line.strip()  # Additional info for previous parameter
                continue
            if line.strip() != "":
                params_lines.append(line.strip())
        elif state == "returns":
            if line.strip() != "":
                returns_line = line.strip()

    parsed_info["description"] = " ".join(description_lines).strip()

    # Parse parameters in the "Args:" section
    for param_line in params_lines:
        param_name, param_payload = parse_param_line(param_line)
        parsed_info["parameters"][param_name] = param_payload

    parsed_info["output_parameter"] = parse_return_line(returns_line)

    return parsed_info


open_api_conversion_dict = {
    "str": "string",
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}
