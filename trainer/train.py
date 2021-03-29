import deepspeed as ds

class DeepSpeedTrain(object):
    def __init__(self, conf):
        self.conf = conf
        self.model = self.build_model()
        pass

    def build_model(self):
        if self.conf['model_type']=='ConformerCTC':
            from model.model import ConformerCTC as Conformer
        elif self.conf['model_type']=='ConformerTransducer':
            from model.model import ConformerTransducer as Conformer

        model = Conformer(self.conf['model_params'])

        pass

    def run(self):