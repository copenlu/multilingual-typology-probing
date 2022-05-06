from typing import Dict, List, Any, Tuple
from utils.setup import load_word_lists_for_language, get_classification_datasets, setup_probe, get_metrics, \
    setup_metrics
import pprint
import json
from tqdm import tqdm
import os
import numpy as np
import wandb


def manual(args):
    if args.wandb:
        tags = [' '.join(args.language), args.trainer, args.attribute, "manual"]
        if args.wandb_tag is not None:
            tags.append(args.wandb_tag)

        config = vars(args)
        config.pop("func", None)
        run = wandb.init(project="flexible-probing-typology", config=config, tags=tags)
        run.name = f"{args.attribute}-{args.language} ({args.trainer}) [{wandb.run.id}]"
        run.save()

    # ::::: SETUP :::::
    print(f"Setting up datasets ({args.language}, {args.attribute}) and probes...")
    word_lists = load_word_lists_for_language(args)
    datasets = get_classification_datasets(args, word_lists, args.attribute)
    dataset_train, dataset_dev, dataset_test, prop_values = datasets["train"], datasets["dev"], datasets["test"], datasets["prop_values"]  # noqa

    setup = setup_probe(args, dataset_train)
    trainer, probe_family = setup["trainer"], setup["probe_family"]

    print("Using validation dataset...")
    eval_metrics = setup_metrics(dataset_dev)

    # ::::: PRE-TRAINING :::::
    # If possible, this pre-trains our model. e.g. using our sampling--based procedure, or by just training
    # the probe once and then zeroing out dimensions we want to ignore. This trainer modifies the
    # neural_probe_model object.
    print("Pre-training probe (if possible)...")
    trainer.train()

    # ::::: EVALUATING THE PROBE :::::
    # We figure out what the parameters (the "specification") of the probe should be, for a subset of
    # dimensions
    select_dimensions = args.dimensions

    print("Probing...")
    print(f"\tSelected dimensions: {select_dimensions or 'ALL'}")

    # The dimensions we want to select
    metrics = get_metrics(trainer=trainer, probe_family=probe_family, metrics=eval_metrics,
                          select_dimensions=select_dimensions)

    if args.wandb:
        wandb.log(metrics)

    # Log to console
    pp = pprint.PrettyPrinter(indent=4)
    print("Results:")
    pp.pprint(metrics)

    # Save to file
    if args.output_file:
        with open(args.output_file, "w") as h:
            json.dump(metrics, h)

        print(f"Saved output to '{args.output_file}'.")


def file(args):
    # ::::: SETUP :::::
    print(f"Setting up datasets ({args.language}, {args.attribute}) and probes...")
    word_lists = load_word_lists_for_language(args)
    datasets = get_classification_datasets(args, word_lists, args.attribute)
    dataset_train, dataset_dev, dataset_test, prop_values = datasets["train"], datasets["dev"], datasets["test"], datasets["prop_values"] # noqa

    setup = setup_probe(args, dataset_train)
    trainer, probe_family = setup["trainer"], setup["probe_family"]

    print("Using validation dataset...")
    eval_metrics = setup_metrics(dataset_dev)

    # ::::: PRE-TRAINING :::::
    # If possible, this pre-trains our model. e.g. using our sampling--based procedure, or by just training
    # the probe once and then zeroing out dimensions we want to ignore. This trainer modifies the
    # neural_probe_model object.
    print("Pre-training probe (if possible)...")
    trainer.train()

    # ::::: EVALUATING THE PROBE :::::
    # We figure out what the parameters (the "specification") of the probe should be, for the current
    # subset of dimensions
    print("Probing...")

    with open(args.file, "r") as h:
        experiments = json.load(h)

    results: List[Dict[str, Any]] = []
    for idx, select_dimensions in tqdm(experiments.items()):
        # The dimensions we want to select
        metrics = get_metrics(trainer=trainer, probe_family=probe_family, metrics=eval_metrics,
                              select_dimensions=select_dimensions)

        # Log to console
        tqdm.write(f"\tExperiment {idx} results: {metrics}")

        results.append({
            "idx": idx,
            "dimensions": select_dimensions,
            "metrics": metrics
        })

    # Save to file
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as h:
            json.dump(results, h)

        print(f"Saved experiment outputs to '{args.output_file}'.")


def greedy(args):
    if args.wandb:
        tags = [' '.join(args.language), args.trainer, args.attribute, "sweep"]
        if args.wandb_tag is not None:
            tags.append(args.wandb_tag)

        config = vars(args)
        config.pop("func", None)
        run = wandb.init(project="flexible-probing-typology", config=config, tags=tags)
        run.name = f"{args.attribute}-{args.language} ({args.trainer}) ({args.selection_criterion}) \
            [{wandb.run.id}]"
        run.save()

    # ::::: SETUP :::::
    print(f"Setting up datasets ({args.language}, {args.attribute}) and probes...")
    word_lists = load_word_lists_for_language(args)
    datasets = get_classification_datasets(args, word_lists, args.attribute)
    dataset_train, dataset_dev, dataset_test, prop_values = datasets["train"], datasets["dev"], datasets["test"], datasets["prop_values"] 

    embedding_dim = dataset_train.get_dimensionality()
    print(f"Embedding dimensionality: {embedding_dim}")

    setup = setup_probe(args, dataset_train, report_progress=True)
    trainer, probe_family = setup["trainer"], setup["probe_family"]

    dimension_selection_metrics = setup_metrics(dataset_dev)
    eval_metrics = setup_metrics(dataset_test)

    # ::::: PRE-TRAINING :::::
    # If possible, this pre-trains our model. e.g. using our sampling--based procedure, or by just training
    # the probe once and then zeroing out dimensions we want to ignore. This trainer modifies the
    # neural_probe_model object.
    print("Pre-training probe (if possible)...")
    trainer.train()

    # ::::: EVALUATING THE PROBE :::::
    # We figure out what the parameters (the "specification") of the probe should be, for the current
    # subset of dimensions
    print("Probing through greedy dimension selection...")

    results: List[Dict[str, Any]] = []
    picked_dimensions = []
    for iteration in range(args.selection_size):
        print(f"Iteration #{iteration}")
        print(f"\tDimensions selected so far: {picked_dimensions}")
        print("\tPicking next dimension...")
        selection_results: List[Tuple[int, float]] = []
        for candidate_dim in tqdm(list(set(range(embedding_dim)) - set(picked_dimensions))):
            metrics_dev = get_metrics(trainer=trainer, probe_family=probe_family,
                                      metrics=dimension_selection_metrics,
                                      select_dimensions=[candidate_dim] + picked_dimensions)
            selection_results.append((candidate_dim, metrics_dev[args.selection_criterion]))

        best_dim = max(selection_results, key=lambda x: x[1])[0]
        picked_dimensions.append(best_dim)

        print(f"\tSelected dimension: {best_dim}")
        metrics_test = get_metrics(trainer=trainer, probe_family=probe_family,
                                   metrics=eval_metrics, select_dimensions=picked_dimensions)

        # Log to console
        print(f"\tIteration results: {metrics_test}")

        results.append({
            "iteration_dimension": best_dim,
            "dimensions": list(picked_dimensions),
            "metrics": metrics_test
        })

        print()

        if args.wandb:
            wandb.log(metrics_test)

    # Add whole vector results & log to wandb
    metrics_test = get_metrics(trainer=trainer, probe_family=probe_family, metrics=eval_metrics,
                               select_dimensions=None)
    results.append({
        "dimensions": "ALL",
        "metrics": metrics_test
    })
    if args.wandb:
        wandb.run.summary.update({f"{k}_ALL": v for k, v in metrics_test.items()})

    # Save to file
    if args.output_file:
        with open(args.output_file, "w") as h:
            json.dump(results, h)

        print(f"Saved experiment outputs to '{args.output_file}'.")