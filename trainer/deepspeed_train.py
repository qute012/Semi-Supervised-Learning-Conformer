import deepspeed as ds
import yaml
import json

from ..scheduler.noam import NoamOpt


class DeepSpeedTrain(object):
    def __init__(self, args):
        self.model_conf = self.build_yaml_conf(args.model_conf)
        self.train_conf = args.train_conf
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
        if self.model_conf.get('model_type') == 'ConformerCTC':
            from model.model import ConformerCTC as Conformer
        elif self.model_conf.get('model_type') == 'ConformerTransducer':
            from model.model import ConformerTransducer as Conformer

        model = Conformer(self.model_conf['model_params'])
        return model

    def build_engine(self):
        self.model, self.optimizer, _, _ = ds.initialize(
            args=self.train_conf,
            model=self.model,
            model_parameters=self.model.parameters(),
            lr_scheduler=NoamOpt
        )

    def run(self):
        pass
