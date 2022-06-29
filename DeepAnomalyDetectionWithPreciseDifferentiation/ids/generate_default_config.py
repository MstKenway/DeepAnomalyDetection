import sys

import yaml

SYSTEM_CONFIG_FILE_NAME = '../system_config.yaml'
PARAM_CONFIG_FILE_NAME = '../param_config.yaml'


def print_usage():
    print("python3 generate_default_config.py [-s](system config) [-p](parameter config)")


def generate_default_system_config(file_name=SYSTEM_CONFIG_FILE_NAME):
    param = {
        "DataSetPath": "dataset",
        "TrainDatasetFileName": "dataset_2_11_little.csv",
        "TestDatasetFileName": "dataset_2_11_little.csv",
        "LogFilePath": "log/log.txt",
        "TrainLogPath": "log/TrainLog",
        "TrainResultPath": "log/TrainLog",
        "TrainedModelPath": "models",
        "TestLogPath": "log/TestLog",
        "TestResultPath": "log/TestLog",
        "Device": "GPU"
    }
    with open(file_name, 'w') as f:
        yaml.dump(param, f)


def generate_default_param_config(file_name=PARAM_CONFIG_FILE_NAME):
    param = {
        "TestName": "FirstTest",
        "DefaultParameter": {
            "TypeName": "GRU",
            "Layer": 2,
            "HiddenSize": 32,
            "DropRate": 0.5,
            "SeqLen": 7,
            "Bidirectional": True,
            "BatchSize": 512,
            "EpochNum": 100,
            "Optimizer": 'Adam',
            "LearningRate": 0.001,
            "ActiveFunction": 'Sigmoid',
        },
        "Specific": [
            {
                "TypeName": "GRU",
            },
        ]
    }
    with open(file_name, 'w') as f:
        yaml.dump(param, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        exit()
    if "-s" in sys.argv:
        generate_default_system_config()
    if "-p" in sys.argv:
        generate_default_param_config()