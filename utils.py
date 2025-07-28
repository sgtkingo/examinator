import os
from glob import glob


def to_bool(value):
    """
    Convert any value to a boolean.

    Args:
        value: The value to convert to a boolean.

    Returns:
        bool: The boolean representation of the value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('true', 'yes', '1'):
            return True
        if value in ('false', 'no', '0', ''):
            return False
        raise ValueError(f"Cannot convert string '{value}' to boolean")
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0

    # For any other types, convert to boolean directly
    return bool(value)


def load_allowed_input_documents(folder: str, allowed_extensions: tuple[str]):
    if os.path.exists(folder):
        documents_list = glob(f'{folder}/*')
        
        documents_list_filtered = [] 

        for file in documents_list:
            if file.endswith(allowed_extensions) and os.path.isfile(file):
                documents_list_filtered.append(file)
        
        if not documents_list_filtered:
            raise Exception('No documents found. Place a document with allowed extension into the input folder!')
        return documents_list_filtered
    raise Exception('The input folder does not exists... You need to be sure that the folder you defined as input exists.')
