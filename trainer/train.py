import deepspeed as ds
import yaml
import json

class DeepSpeedTrain(object):
    def __init__(self, conf):
        self.conf = conf
        self.model = self.build_model()
        self.build_engine()

    def build_yaml_conf(self, conf_path):
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)
        return conf

    def build_json_conf(self, conf_path):
        with open(conf_path, 'r') as f:
            conf = json.loads(f)
        return conf

    def build_model(self):
        if self.conf['model_type']=='ConformerCTC':
            from model.model import ConformerCTC as Conformer
        elif self.conf['model_type']=='ConformerTransducer':
            from model.model import ConformerTransducer as Conformer

        model = Conformer(self.conf['model_params'])
        return model

    def build_engine(self):
        self.model, self.optimizer

    def run(self):
        pass