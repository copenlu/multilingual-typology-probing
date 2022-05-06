from typing import List, Union, Dict, Type, Any
import pickle
from tqdm import tqdm
import torch
from pathlib import Path
import yaml
import warnings
import random
import os

from probekit.utils.types import Word
from probekit.utils.dataset import WordClassificationDataset
from probekit.models.probe import Probe
from probekit.trainers.trainer import Trainer
from probekit.metrics.metric import Metric
from probekit.metrics.mutual_information import MutualInformation
from probekit.metrics.accuracy import Accuracy
from probekit.models.discriminative.neural_probe import NeuralProbe
from probekit.trainers.gaussian_probe_trainer import GaussianProbeTrainer
from probekit.models.generative.gaussian_probe import GaussianProbe

from trainers.poisson_trainer import PoissonTrainer
from trainers.conditional_poisson_trainer import ConditionalPoissonTrainer
from trainers.simple_trainer import SimpleTrainer
from models.neural_probe import MLPProbeModel


def get_config():
    with open("config.yml", "r") as h:
        config = yaml.load(h, Loader=yaml.FullLoader)

    return config

def get_attributes():
    with open("scripts/properties.lst", "r") as props:
        properties = props.readlines()

    return [prop.strip("\n") for prop in properties]

def get_languages():
    with open("scripts/languages_common.lst", "r") as langs:
        languages = langs.readlines()

    return [lang.strip("\n") for lang in languages]


def convert_pickle_to_word_list(path: Union[Path, str]) -> List[Word]:
    # Load pickle file
    with open(path, "rb") as h:
        data = pickle.load(h)

    # Convert data to example collection
    word_list: List[Word] = []
    for w in tqdm(data):
        word_list.append(
            Word(w["word"], torch.tensor(w["embedding"]).squeeze(0), w["attributes"]))
    return word_list


def load_word_lists_for_language(args, test_branch = False) -> Dict[str, List[Word]]:
    # Read files and obtain word list
    config = get_config()
    data_root = Path(config["data"]["datasets_root"])
    languages = get_languages()

    train: List[Word] = []
    dev: List[Word] = []
    test: List[Word] = []


    language = args.test_language if test_branch else args.language
    if isinstance(language, str):
        queued_languages = [language]
    else:
        queued_languages = language

    for lang in queued_languages:
        
        if lang not in languages:
            raise ValueError(f"Invalid language choice '{lang}'. Note that this has changed since "
                             "last version, and this should be the name of a UD treebank. Valid choices: "
                             f"{languages}")

        treebank = lang

        embedding = args.embedding
        treebank_path = data_root / treebank
        file_path_train = next(treebank_path.glob(f"*-train-{embedding}.pkl"))
        file_path_dev = next(treebank_path.glob(f"*-dev-{embedding}.pkl"))
        file_path_test = next(treebank_path.glob(f"*-test-{embedding}.pkl"))

        train.extend(convert_pickle_to_word_list(file_path_train))
        dev.extend(convert_pickle_to_word_list(file_path_dev))
        test.extend(convert_pickle_to_word_list(file_path_test))

        # Shuffle data
        random.shuffle(train)
        random.shuffle(dev)
        random.shuffle(test)

    return {
        "train": train,
        "dev": dev,
        "test": test
    }

def get_classification_datasets(
        args, word_lists: Dict[str, List[Word]], attribute: str) -> Dict[str, WordClassificationDataset]:
    device = "cuda:0" if args.gpu else "cpu"
    words_train, words_dev, words_test = word_lists["train"], word_lists["dev"], word_lists["test"]

    # These attributes, by default, have spaces but this just complicates a lot of file management.
    # So we replace attributes with spaces with others.
    if attribute not in get_attributes():
        raise ValueError("The provided attribute is not ")

    if attribute == "POS":
        attribute = "Part of Speech"
    elif attribute == "ArgumentMark":
        attribute = "Argument Marking"
    elif attribute == "Gender":
        attribute = "Gender and Noun Class"
    elif attribute == "InfoStructure":
        attribute = "Information Structure"
    elif attribute == "Switch-Reference":
        attribute = "SwitchRef"

    # Figure out list of all values, for this property, that meet the minimum threshold of 100 instances
    min_count = 20
    property_value_list = WordClassificationDataset.get_property_value_list(
        attribute, words_train, words_dev, words_test, min_count=min_count)

    if len(property_value_list) <= 1:
        warnings.warn(f"Not enough classes to run experiment. This attribute has {len(property_value_list)}"
                      f" classes with at least {min_count} examples across all splits (need at least 2). "
                      "Skipping...")
        exit()

    # Convert word list into WordClassificationDataset. This is the type of the datasets expected by the tool
    # (it's just a dictionary, where the keys are the values of the property, e.g., "Singular", "Plural",
    # and the values are a list of training examples)
    dataset_train = WordClassificationDataset.from_unit_list(words_train, attribute=attribute, device=device,
                                                             property_value_list=property_value_list)
    dataset_dev = WordClassificationDataset.from_unit_list(words_dev, attribute=attribute, device=device,
                                                           property_value_list=property_value_list)
    dataset_test = WordClassificationDataset.from_unit_list(words_test, attribute=attribute, device=device,
                                                            property_value_list=property_value_list)

    return {
        "train": dataset_train,
        "dev": dataset_dev,
        "test": dataset_test,
        "prop_values": property_value_list
    }


def get_metrics(trainer: Trainer, probe_family: Type[Probe], metrics: Dict[str, Metric],
                select_dimensions: List[int]) -> Dict[str, float]:
    device = trainer.get_device()
    specification = trainer.get_specification(select_dimensions)

    # We instantiate the probe from the specification we just obtained.
    # This allows us to actually use it to, e.g., evaluate our metrics.
    probe = probe_family.from_specification(specification, device=device)

    # Compute & report metrics
    return {metric_name: metric.compute(probe=probe, select_dimensions=select_dimensions)
            for metric_name, metric in metrics.items()}


def setup_metrics(eval_dataset: WordClassificationDataset) -> Dict[str, Metric]:
    return {
        "mi": MutualInformation(dataset=eval_dataset, bits=True, normalize=True),
        "accuracy": Accuracy(dataset=eval_dataset)
    }


def setup_probe(args, dataset_train: WordClassificationDataset, report_progress: bool = True) -> Dict[str, Any]:
    # We use a general neural probe
    device = "cuda:0" if args.gpu else "cpu"
    embedding_size = 1024 if args.embedding == "xlm-roberta-large" else 768
    neural_probe_model = MLPProbeModel(
        embedding_size=embedding_size, num_classes=len(dataset_train.keys()),
        hidden_size=args.probe_num_hidden_units, num_layers=args.probe_num_layers,
        activation = args.activation, decomposable=args.decomposable).to(device)

    shared_args = dict(
        model=neural_probe_model, dataset=dataset_train, device=device,
        num_epochs=args.trainer_num_epochs, report_progress=report_progress, patience=args.patience,
        l1_weight=args.l1_weight, l2_weight=args.l2_weight, batch_size=args.batch_size,
        lr=args.learning_rate
    )

    if args.trainer in ["upperbound", "lowerbound"]:
        decomposable = False if args.trainer == "upperbound" else True
        trainer = SimpleTrainer(decomposable=decomposable, **shared_args)
        probe_family: Type[Probe] = NeuralProbe
    elif args.trainer == "poisson":
        trainer = PoissonTrainer(
            mc_samples=args.mc_samples, entropy_scale=args.entropy_scale, 
            temperature=args.temperature, temp_annealing=args.temp_annealing,
            temp_min=args.temp_min, anneal_rate=args.anneal_rate, **shared_args)
        probe_family = NeuralProbe
    elif args.trainer == "conditional-poisson":
        trainer = ConditionalPoissonTrainer(
            mc_samples=args.mc_samples,
            temperature=args.temperature, temp_annealing=args.temp_annealing,
            temp_min=args.temp_min, anneal_rate=args.anneal_rate, **shared_args)
        probe_family = NeuralProbe
    elif args.trainer == "qda":
        trainer = GaussianProbeTrainer(mode="map", dataset=dataset_train, device=device)
        probe_family = GaussianProbe
    else:
        raise Exception("Unknown trainer")

    return {
        "trainer": trainer,
        "probe_family": probe_family,
    }
