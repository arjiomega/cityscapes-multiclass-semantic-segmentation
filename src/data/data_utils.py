def extract_data_name(filename: str) -> str:
    """Removes additional information from the filename.
    Example:
        'img_000000_000000_leftImg8bit.png' -> 'img_000000_000000'
    """
    str2list = filename.split("_")[:3]
    return "_".join(str2list)
