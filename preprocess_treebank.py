import collections
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any
import numpy as np

import torch
from torch_scatter import scatter_mean
from tqdm import tqdm
from conllu import parse_incr, TokenList
import os
from os import path
import yaml
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel
import pickle
from argparse import ArgumentParser
from utils.parser import parse_unimorph_features
import config
import json


_DEFAULT_TREEBANKS_ROOT = path.join(config.DATA_ROOT, "ud/ud-treebanks-v2.1")

parser = ArgumentParser()
parser.add_argument("treebank", type=str)  # e.g. "UD_Portuguese-Bosque"
parser.add_argument("--treebanks-root", type=str, default=_DEFAULT_TREEBANKS_ROOT)
parser.add_argument("--dry-run", default=False, action="store_true", help="If enabled, will not actually \
                    compute any embeddings, but go over the dataset and do everything else.")
parser.add_argument("--bert", default=None)
parser.add_argument("--xlmr", default=None)
parser.add_argument("--use-gpu", action="store_true", default=False)
parser.add_argument("--skip-existing", action="store_true", default=False)
args = parser.parse_args()


if not (args.bert or args.xlmr):
    raise Exception("Must do either BERT or XLMR, but not more than one")

treebank_path = os.path.join(args.treebanks_root, args.treebank)
limit_number = None
bert_model = args.bert
xlmr_model = args.xlmr
print("Embeddings root:", config.EMBEDDINGS_ROOT)

skip_existing = args.skip_existing
device = 'cpu'
if args.use_gpu:
    print("Using GPU")
    device = 0


def subword_tokenize(tokenizer, tokens: List[str]) -> List[Tuple[int, str]]:
    """
    Returns: List of subword tokens, List of indices mapping each subword token to one real token.
    """
    subtokens = [tokenizer.tokenize(t) for t in tokens]

    indexed_subtokens = []
    for idx, subtoks in enumerate(subtokens):
        for subtok in subtoks:
            indexed_subtokens.append((idx, subtok))

    return indexed_subtokens


def unimorph_feature_parser(line: List[str], i: int) -> Dict[str, str]:
    if line[i] == "_":
        return {}

    return parse_unimorph_features(line[i].split(";"))


def merge_attributes(tokens: List[str], value_to_attr_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Returns a dictionary containing Unimorph attributes, and the values taken on after the merge.
    """
    # First, build a list that naively merges everything
    merged_attributes: Dict[str, List[str]] = {}
    for t in tokens:
        for attr, val in t["um_feats"].items():
            if attr not in merged_attributes:
                merged_attributes[attr] = []

            merged_attributes[attr].append(val)

    # Second, remove attributes with multiple values (even if they are the same)
    final_attributes: Dict[str, str] = {}
    for attr, vals in merged_attributes.items():
        if len(vals) == 1:
            final_attributes[attr] = vals[0]

    return final_attributes


# Setup debugging tracker
total = 0
skipped: Dict[str, int] = {}

final_token_list: List[TokenList] = []

for f in os.listdir(treebank_path):
    if path.isfile(path.join(treebank_path, f)) and "-um-" in f and f.endswith(".conllu"):
        filename = f
        full_path = path.join(treebank_path, filename)
        

        # Load possible UM tags
        tags_file = "utils/tags.yaml"
        with open(tags_file, 'r') as h:
            _UNIMORPH_ATTRIBUTE_VALUES = yaml.full_load(h)["categories"]

        _UNIMORPH_VALUES_ATTRIBUTE = {v: k for k, vs in _UNIMORPH_ATTRIBUTE_VALUES.items() for v in vs}
        # Setup UM feature parsing
        _FEATS = ["id", "form", "lemma", "upos", "xpos", "um_feats", "head", "deprel", "deps", "misc"]

        # Parse Conll-U files with UM
        with open(full_path, "r") as h:
            # Setup BERT tokenizer here provisionally as we need to know which sentences have 512+ subtokens
            if args.xlmr:
                tokenizer = XLMRobertaTokenizer.from_pretrained(args.xlmr)
            else:
                tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

            for sent_id, tokenlist in enumerate(tqdm(
                    parse_incr(h, fields=_FEATS, field_parsers={"um_feats": unimorph_feature_parser}))):
                # Only process first `limit_number` if it is set
                if limit_number is not None and sent_id > limit_number:
                    break

                # Remove virtual nodes
                tokenlist = [t for t in tokenlist if not (isinstance(t["id"], tuple) and t["id"][1] == ".")]

                # Build list of ids that are contracted
                contracted_ids: List[int] = []
                for t in tokenlist:
                    if isinstance(t["id"], tuple):
                        if t["id"][1] == "-":
                            # Range
                            contracted_ids.extend(list(range(t["id"][0], t["id"][2] + 1)))

                # Build dictionary of non-contracted token ids to tokens
                non_contracted_token_dict: Dict[int, Any] = {
                    t["id"]: t for t in tokenlist if not isinstance(t["id"], tuple)}

                # Build final list of (real) tokens, without any contractions
                # Contractions are assigned the attributes of the constituent words, unless there is a clash
                # with one attribute taking more than one value (e.g. POS tag is a frequent example), whereby
                # we discard it.

                # Filter only for content words if parameter is set to true
                final_tokens: List[Any] = []
                for t in tokenlist:
                    if isinstance(t["id"], tuple):
                        constituent_ids = list(range(t["id"][0], t["id"][2] + 1))
                        t["um_feats"] = merge_attributes(
                            [non_contracted_token_dict[x] for x in constituent_ids],
                            _UNIMORPH_VALUES_ATTRIBUTE)

                        # If this is a contraction, add it
                        final_tokens.append(t)

                    elif t["id"] not in contracted_ids:
                        # Check if this t is part of a contraction
                        final_tokens.append(t)

                final_tokens: TokenList = TokenList(final_tokens)

                # Skip if this would have more than 512 subtokens
                labelled_subwords = subword_tokenize(tokenizer, [t["form"] for t in final_tokens])
                subtoken_indices, subtokens = zip(*labelled_subwords)
                if len(subtokens) >= 512:
                    if "subtoken_count" not in skipped:
                        skipped["subtoken_count"] = 0

                    skipped["subtoken_count"] += 1
                    continue

                if "total_sents" not in skipped:
                    skipped["total_sents"] = 0

                skipped["total_sents"] += 1

                # Add this sentence to the list we are processing
                final_token_list.append(final_tokens)

                if args.dry_run:
                    print("Dry run finished.")
                    continue

# Print logs:
print("Skipped:")
print(skipped)
print()

print(f"Total: {total}")

final_results = []
if args.bert:
    model_name = bert_model
    print(f"Processing {args.treebank}...")

    # Setup BERT
    model = BertModel.from_pretrained(bert_model).to(device)

    # Subtokenize, keeping original token indices
    results = []
    for sent_id, tokenlist in enumerate(tqdm(final_token_list)):
        labelled_subwords = subword_tokenize(tokenizer, [t["form"] for t in tokenlist])
        subtoken_indices, subtokens = zip(*labelled_subwords)
        subtoken_indices_tensor = torch.tensor(subtoken_indices).to(device)

        # We add special tokens to the sequence and remove them after getting the BERT output
        subtoken_ids = torch.tensor(
            tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(subtokens))).to(device)

        results.append((tokenlist, subtoken_ids, subtoken_indices_tensor))

    # Prepare to compute BERT embeddings
    model.eval()

    # NOTE: No batching, right now. But could be worthwhile to implement if a speed-up is necessary.
    for token_list, subtoken_ids, subtoken_indices_tensor in tqdm(results):
        total += 1

        with torch.no_grad():
            # shape: (batch_size, max_seq_length_in_batch + 2)
            inputs = subtoken_ids.reshape(1, -1)

            # shape: (batch_size, max_seq_length_in_batch)
            indices = subtoken_indices_tensor.reshape(1, -1)

            # shape: (batch_size, max_seq_length_in_batch + 2, embedding_size)
            outputs = model(inputs)
            final_output = outputs[0]

            # shape: (batch_size, max_seq_length_in_batch, embedding_size)
            # Here we remove the special tokens (BOS, EOS)
            final_output = final_output[:, 1:, :][:, :-1, :]

            # Average subtokens corresponding to the same word
            # shape: (batch_size, max_num_tokens_in_batch, embedding_size)
            token_embeddings = scatter_mean(final_output, indices, dim=1)

        # Convert to python objects
        embedding_list = [x.cpu().numpy() for x in token_embeddings.squeeze(0).split(1, dim=0)]

        for t, e in zip(token_list, embedding_list):
            t["embedding"] = e

        final_results.append(token_list)

elif args.xlmr:
    output_filename = filename.split('.')[0] + f"{args.xlmr}.pkl"
    output_file = path.join(treebank_path, output_filename)
    model_name = args.xlmr

    print(f"Processing {filename}...")

    # Setup XLM-R
    model = XLMRobertaModel.from_pretrained(model_name).to(device)

    # Subtokenize, keeping original token indices
    results = []
    for sent_id, tokenlist in enumerate(tqdm(final_token_list)):
        labelled_subwords = subword_tokenize(tokenizer, [t["form"] for t in tokenlist])
        subtoken_indices, subtokens = zip(*labelled_subwords)
        subtoken_indices_tensor = torch.tensor(subtoken_indices).to(device)

        # We add special tokens to the sequence and remove them after getting the output
        subtoken_ids = torch.tensor(
            tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(subtokens))).to(device)

        results.append((tokenlist, subtoken_ids, subtoken_indices_tensor))

    # Prepare to compute embeddings
    model.eval()

    # NOTE: No batching, right now. But could be worthwhile to implement if a speed-up is necessary.
    for token_list, subtoken_ids, subtoken_indices_tensor in tqdm(results):
        total += 1

        with torch.no_grad():
            # shape: (batch_size, max_seq_length_in_batch + 2)
            inputs = subtoken_ids.reshape(1, -1)

            # shape: (batch_size, max_seq_length_in_batch)
            indices = subtoken_indices_tensor.reshape(1, -1)

            # shape: (batch_size, max_seq_length_in_batch + 2, embedding_size)
            outputs = model(inputs)
            final_output = outputs[0]

            # shape: (batch_size, max_seq_length_in_batch, embedding_size)
            # Here we remove the special tokens (BOS, EOS)
            final_output = final_output[:, 1:, :][:, :-1, :]

            # Average subtokens corresponding to the same word
            # shape: (batch_size, max_num_tokens_in_batch, embedding_size)
            token_embeddings = scatter_mean(final_output, indices, dim=1)

        # Convert to python objects
        embedding_list = [x.cpu().numpy() for x in token_embeddings.squeeze(0).split(1, dim=0)]

        for t, e in zip(token_list, embedding_list):
            t["embedding"] = e

        final_results.append(token_list)

# Keep important parts
final_results_filtered = []
for row in final_results:
    for token in row:
        
        final_results_filtered.append({
            "word": token["form"],
            "lemma": token["lemma"],
            "embedding": token["embedding"],
            "attributes": token["um_feats"],
        })

# Recreate data set split so that a word comes up only in one of the data sets
unique_words = dict(collections.Counter(t["lemma"] for t in final_results_filtered))
word_types = dict(list(unique_words.items()))

if len(word_types) <= 10:
    raise Exception("Not enough words to form splits")

# Train test split to shuffles the word and split. We use a 80/10/10
train_words, dev_test_words = train_test_split(list(unique_words.keys()), test_size=0.2, random_state=0)
dev_words, test_words = train_test_split(dev_test_words, test_size=0.5, random_state=0)

# Optimize inclusion checks
train_words, dev_words, test_words = set(train_words), set(dev_words), set(test_words)

print("Building final...")
train = [t for t in final_results_filtered if t['lemma'] in train_words]
dev = [t for t in final_results_filtered if t['lemma'] in dev_words]
test = [t for t in final_results_filtered if t['lemma'] in test_words]

print(f"Final sizes: {len(train)}/{len(dev)}/{len(test)}")

# Save final results
print("Save data sets")

train_file = path.join(treebank_path, "{}-train-{}.pkl".format(args.treebank, model_name))
test_file = path.join(treebank_path, "{}-test-{}.pkl".format(args.treebank, model_name))
dev_file = path.join(treebank_path, "{}-dev-{}.pkl".format(args.treebank, model_name))

with open(train_file, "wb") as h:
    pickle.dump(train, h)

with open(test_file, "wb") as h:
    pickle.dump(test, h)

with open(dev_file, "wb") as h:
    pickle.dump(dev, h)
