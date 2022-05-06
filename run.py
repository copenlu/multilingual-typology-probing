import argparse

import commands


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(description="General script to run experiments. Different modes are \
                                   available depending on what type of experiment you want to run. The \
                                   options are split into two groups: general options (shown below), and \
                                   mode-specific options.")

parser.add_argument("--language", type=str, nargs="+", required=True, help="The three-letter code for the language \
                    or multiple languages you want to probe (e.g., eng). Should be the same as used by Unimorph.",
                    default=argparse.SUPPRESS)
parser.add_argument("--attribute", type=str, required=True, help="The attribute (aka. Unimorph dimension) \
                    to be probed (e.g., \"Number\", \"Gender and Noun Class\").", default=argparse.SUPPRESS)
parser.add_argument("--trainer", choices=["fixed", "upperbound", "lowerbound", "qda", "poisson",
                    "conditional-poisson"],
                    required=True, default=argparse.SUPPRESS,
                    help="The type of trainer you want to use. Fixed is the one introduced in the paper.\
                    Lowerbound trains the probe once and then masks out dimensions naively, whereas \
                    upperbound re-trains the probe for every evaluated set of dimensions. QDA is the probe \
                    introduced in \"Intrinsic Probing through Dimension Selection\".")
parser.add_argument("--gpu", default=False, action="store_true", help="Pass this flag if you want to use a \
                    GPU to speed up the experiments. Multiple GPUs are not supported.")
parser.add_argument("--embedding", type=str, choices=["bert-base-multilingual-cased", "xlm-roberta-base",
                    "xlm-roberta-large"], default="bert-base-multilingual-cased", 
                    help="Type of embedding, either bert or xlmr.")
parser.add_argument("--trainer-num-epochs", type=int, default=5000, help="The maximum number of epochs that \
                    probes should be trainer for.")
parser.add_argument("--probe-num-layers", type=int, default=1, help="The number of layers the probe should \
                    have. If this is set to 1, uses a logistic sigmoid probe.")
parser.add_argument("--activation", type=str, default="sigmoid", help="The activation function \
                    in the MLP.")
parser.add_argument("--probe-num-hidden-units", type=int, default=50, help="The number of hidden units in \
                    each layer of the probe.")
parser.add_argument("--output-file", type=str, help="If provided, results of the experiment will be written \
                    to this file in JSON format.")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter in Gumbel \
                    softmax.")
parser.add_argument("--temp-annealing", type=bool, default=False, help="Pass this flag if you want to use \
                    temperature annealing in Gumbel softmax.")
parser.add_argument("--temp-min", type=float, default=0.5, help="The minimum value for the temperature \
                    parameter in Gumbel softmax.")
parser.add_argument("--anneal-rate", type=float, default=0.00003, help="The anneal rate for the temperature \
                    parameter in Gumbel softmax.")
parser.add_argument("--patience", type=int, default=50, help="The number of epochs after which the training \
                    will stop if the loss has not decreased.")
parser.add_argument("--l1-weight", type=float, default=0.0, help="L1 regularization tuning hyperparameter.")
parser.add_argument("--l2-weight", type=float, default=0.0, help="L2 regularization tuning hyperparameter.")
parser.add_argument("--mc-samples", type=int, default=5, help="Number of MC samples.")
parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--entropy-scale", type=float, default=1e-2, help="Entropy scale")
parser.add_argument("--decomposable", action="store_true", default=False, help="If set, makes the probe's \
                    first layer decomposable.")
parser.add_argument("--wandb", default=False, action="store_true", help="If enabled, logs runs to the \
                    appropriate Weights & Biases project.")
parser.add_argument("--wandb-tag", type=str, help="This can be used to set a tag on the Weights & Biases \
                    run (e.g., for easier filtering down the road).")

# ::::: MANUAL MODE :::::
# You can manually specify the dimensions you want to check--mostly for development and debugging purposes.
parser_manual = subparsers.add_parser("manual", description="In manual mode, probe performance is evaluated \
                                      for a pre-specified set of dimensions.")
parser_manual.add_argument("--dimensions", type=int, nargs="+", help="The dimensions to be probed. Note that \
                           these, by convention, are zero-indexed.")
parser_manual.set_defaults(func=commands.manual)

# ::::: FILE MODE :::::
# Dimensions are specified from a file
parser_file = subparsers.add_parser("file", description="In file mode, probe performance is evaluated for a \
                                    whole host of configurations. This can be thought of as repetitively \
                                    calling manual mode for different dimensions (with some additional \
                                    optimizations), where the configurations are stored in a JSON file.")
parser_file.add_argument("--file", type=str, required=True, help="The file containing the configurations to \
                         be probed.")
parser_file.set_defaults(func=commands.file)

# ::::: GREEDY MODE :::::
# Select dimensions greedily using dev set
parser_greedy = subparsers.add_parser("greedy", description="In greedy mode, greedy dimension selection is \
                                      performed, like in \"Intrinsic Probing through Dimension Selection\".")
parser_greedy.add_argument("--selection-size", type=int, default=50, help="The number of dimensions to be \
                           selected.")
parser_greedy.add_argument("--selection-criterion", type=str, choices=["mi", "accuracy"], required=True,
                           help="The metric that should be used for dimension seleciton.")

parser_greedy.set_defaults(func=commands.greedy)

args = parser.parse_args()

args.func(args)
