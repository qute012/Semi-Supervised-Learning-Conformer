import yaml


def build_conf(conf_path):
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    return conf
