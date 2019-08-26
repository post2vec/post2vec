def load_json(json_fpath):
    import json

    with open(json_fpath) as f:
        data = json.load(f)

    return data
