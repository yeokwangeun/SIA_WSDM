# Sequential recommendation with Iterative Attention on content Features

## Environmental Setup
1. Create a conda environment with python 3.8:
```bash
conda create -n {env_name} python=3.8
conda activate {env_name}
```

2. Install the required packages usinng the 'requirements.txt' file:
```bash
conda install - file requirements.txt
```

## Dataset

To download a dataset, run the following script:

```bash
bash download_dataset.sh {dataset_name}
```

Replace {dataset_name} with the name of one of the available datasets listed below:
- amazon_beauty
- amazon_toys
- amazon_sports
- ml-1m

for example:

```bash
bash download_dataset.sh amazon_beauty
```

## Usage
You can run main.py with various options. For more detailed information about the arguments, you can check by running the command:

```bash
python main.py --help
```

To use the best hyperparameters for each dataset, run the main.py script using the provided YAML file. Here's an example of how to use it for the amazon_beauty dataset:

```bash
python main.py --config_file config/amazon_beauty.yaml
```