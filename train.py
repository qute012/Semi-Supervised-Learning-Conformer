import argparse

from utils.utils import build_conf
from trainer.train import DeepSpeedTrain


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End Speech Recognition Training'
    )
    parser.add_argument(
        '--conf',
        default='config/conformer_m.yaml',
        help='configuration path for training',
        type=str,
    )
    args = parser.parse_args()
    conf = build_conf(args.conf)
    DeepSpeedTrain(conf).run()


if __name__ == '__main__':
    main()
