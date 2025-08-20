class Constants:
    """
    This is just simply loading the Constants from the json file.
    """
    def __init__(self, path="constants.json"):
        import json
        with open(path, "r") as f:
            self._data = json.load(f)

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"No constant '{key}'")
    
    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"No constant '{key}'")