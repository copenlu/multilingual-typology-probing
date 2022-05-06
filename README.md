# Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models

This repository contains code accompanying the paper: [Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models (Stańczak et al., NAACL 2022)](https://arxiv.org/abs/2205.02023).

## Setup

These instructions assume that conda is already installed on your system.

1. Clone this repository. *NOTE: We recommend keeping the default folder name when cloning.*
2. First run `conda env create -f environment.yml`.
3. Activate the environment with `conda activate multilingual-typology-probing`.
4. (Optional) Setup wandb, if you want live logging of your runs.

### Generate data

You will also need to generate the data.

1. First run `mkdir unimorph && cd unimorph && wget https://raw.githubusercontent.com/unimorph/um-canonicalize/master/um_canonicalize/tags.yaml`
2. Download [UD 2.1 treebanks](https://universaldependencies.org/) and put them in `data/ud/ud-treebanks-v2.1`
3. Clone the modified [UD converter](git@github.com:ltorroba/ud-compatibility.git) to this repo's parent folder and then convert the treebank annotations to the UniMorph schema `./scripts/ud_to_um.sh`.
4. Run `./scripts/preprocess_bert.sh`, `./scripts/preprocess_xlmr_base.sh`, and `./scripts/preprocess_xlmr_large.sh` to preprocess all the relevant treebanks using relevant embeddings. This may take a while.

## Run Experiments

The `run.py` script can be used to invoke the experiments.
Commands are of the format `python run.py [ARGS] MODE [MODE-ARGS]`.

First, in our paper we employ a latent variable probe presented in [A Latent-Variable Model for Intrinsic Probing (Stańczak et al., 2022)](https://arxiv.org/abs/2201.08214) to identify the relevant subset of neurons in each language for each morphosyntactic attribute. We opt for a Poisson sampling scheme. We solve the optimization problem using greedy search using mutual information as a performance measure. 

Hence, we run `python run.py --language $language --attribute $attribute --trainer poisson --gpu --embedding $embedding greedy --selection-size 50 --selection-criterion mi` for each analysed language--attribute pair for each of the three probed language models, m-BERT, XLM-R-base, and XLM-R-large. 
Alternatively, you can also run `make 01_bert_ALL`, `make 01_xlmr_base_ALL`, and `make 01_xlmr_large_ALL` which run the above command for all the chosen languages and attributes, and generates the appropriate files.

Next, you can run exploratory analysis on the generated results. Plots presented in the paper were generated with the following files: `01_neuron_overlap.py`, `02_lang_similarity_no_attr.py`, and `03_genus_similarity.py`.


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:


```bash
@inproceedings{stanczak-etal-2022-same,
    title = "Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models",
    author = "Stańczak, Karolina and 
    Ponti, Edoardo and 
    Torroba Hennigen, Lucas and 
    Cotterell, Ryan and 
    Augenstein, Isabelle",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2205.02023",
}
```

#### Contact

To ask questions or report problems, please open an [issue](https://github.com/copenlu/multilingual-typology-probing/issues).
