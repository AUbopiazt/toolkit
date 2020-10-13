import json


def get_cfg_from_json(cfg, test_set):
    with open(cfg, 'r') as fd:
        cfg_dct = json.load(fd)
    return cfg_dct[test_set]
