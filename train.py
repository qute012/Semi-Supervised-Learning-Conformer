import argparse

from trainer.deepspeed_train import DeepSpeedTrain


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End Speech Recognition Training'
    )
    parser.add_argument(
        '--model_conf',
        default='config/conformer_m.yaml',
        help='model configuration path for training',
        type=str,
    )
    parser.add_argument(
        '--train_conf',
        default='config/conformer_m.yaml',
        help='training configuration path for training',
        type=str,
    )

    args = parser.parse_args()
    DeepSpeedTrain(args).run()


if __name__ == '__main__':
    main()
