import yaml


def get_config(config_path = 'config.yaml'):
    with open(config_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile,Loader=yaml.Loader)
    return cfg
