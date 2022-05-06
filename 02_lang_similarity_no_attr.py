import json
from typing import Dict, List, Tuple, Set, Optional, Any
from os import listdir
from os.path import isfile, join

import plotly.graph_objects as go
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import random
from pathlib import Path

from generate_graphs_and_tables import lang_sorted_family
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns
import pycountry

from lang2vec import lang2vec as l2v

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


parser = ArgumentParser()
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--show-plot", default=False, action="store_true")
parser.add_argument("--random-baseline", default=False, action="store_true")
args = parser.parse_args()

top_k = args.top_k

"""
cache_file = f"runs-{tag}-selected-dims.pkl"
# Create cache
results_raw: List[Tuple[Dict[str, Any], List[int]]] = []
if not os.path.exists(cache_file):
    api = wandb.Api()
    runs = api.runs("ltorroba/interp-bert", {"tags": {"$in": [tag]}})
    print("Found %i" % len(runs))

    for run in tqdm(runs):
        run.file("results.json").download(replace=True)

        dims: List[int] = []
        with open("results.json", "r") as h:
            results = json.load(h, cls=ResultsDecoder)

            for r in results:
                dims.append(r["candidate_dim"])

            print(run.name, dims)

        assert len(dims) == 50

        results_raw.append((run.config, dims))

    with open(cache_file, "wb") as h:
        pickle.dump(results_raw, h)
else:
    with open(cache_file, "rb") as h:
        results_raw = pickle.load(h)
"""

file_list_bert = [f for f in listdir("results/01_bert_results/")
            if isfile(join("results/01_bert_results/", f)) and ".json" in f]
file_list_xlmr = [f for f in listdir("results/01_xlmr_results/")
            if isfile(join("results/01_xlmr_results/", f)) and ".json" in f]
file_list = file_list_bert + file_list_xlmr

RESULTS = []

rel_treebanks = []
with open("scripts/languages_common.lst", "r") as h:
    for l in h:
        rel_treebank = l.strip("\n")
        rel_treebanks.append(rel_treebank)

for f in file_list:
    match = f.split("---")
    l = match[0]
    a = match[1]
    e = match[2].split(".")[0]

    if l in rel_treebanks:
        RESULTS.append((l, a, e))

RESULTS = [(l, a, e) for (l, a, e) in RESULTS]
attributes = set(item[1] for item in RESULTS)

def convert_language_code(treebank_name):
    """ Converts treebank names to language codes. """
    lang_name = treebank_name[3:].split("-")[0].replace("_", " ")
    lang = pycountry.languages.get(name=lang_name)

    if lang is not None:
        return lang.alpha_3.lower()

    return "unk"

# embedding, attribute, language
results_raw: List[Tuple[Dict[str, Any], List[int]]] = []
for l, a, e in RESULTS:  # noqa
    embedding_size = 1024 if e == "xlm-roberta-large" else 768
    DEFAULT_RESULTS_FOLDER = "results/01_bert_results/" if e == "bert-base-multilingual-cased" else "results/01_xlmr_results/"
    DEFAULT_FILE_FORMAT = DEFAULT_RESULTS_FOLDER + "{lang}---{attribute}---{embedding}.json"
    with open(DEFAULT_FILE_FORMAT.format(lang=l, attribute=a, embedding=e), "r") as h:
        data = json.load(h)
    results_raw.append(
        (
            { "embedding": e, "attribute": a,
            "language": convert_language_code(l) },
            [d["iteration_dimension"] for d in data if "iteration_dimension" in d]
            if not args.random_baseline else random.sample(range(embedding_size), k=args.top_k)
        )
    )


def compute_overlap(data_raw, top_k):
    mark_count: Dict[str, int] = {}  # num of languages logging that attribute
    results: Dict[str, Counter] = {}
    for run_config, dims in data_raw:
        if run_config["embedding"] != embedding:
            continue

        # Increment mark counting
        if run_config["attribute"] not in mark_count:
            mark_count[run_config["attribute"]] = 0

        mark_count[run_config["attribute"]] += 1

        # Increment actual counters
        if run_config["attribute"] not in results:
            results[run_config["attribute"]] = Counter()

        results[run_config["attribute"]].update(dims[:top_k])

    return results, mark_count


def compute_similarity_for_attribute(attribute, embedding, data_raw, top_k, language_order: Optional[List[str]] = None):
    data_list: List[Tuple[str, Set[int]]] = []
    for run_config, dims in data_raw:
        if run_config["embedding"] != embedding:
            continue

        if run_config["attribute"] != attribute:
            continue

        data_list.append((run_config["language"], set(dims[:top_k])))

    if not language_order:
        data_list = sorted(data_list, key=lambda x: x[0])
        return [x[0] for x in data_list], compute_similarity(data_list)
    else:
        data_list_dict = {k: v for k, v in data_list}
        data_list_sorted = []
        for x in language_order:
            if x in data_list_dict:
                data_list_sorted.append((x, data_list_dict[x]))

        data_list = data_list_sorted
        return [x[0] for x in data_list], compute_similarity(data_list)

def compute_jaccard_index(set_a: Set[int], set_b: Set[int]) -> float:
    return len(set_a & set_b) / len(set_a | set_b)


def compute_overlap_coefficient(set_a: Set[int], set_b: Set[int]) -> float:
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def compute_similarity(data_list: List[Tuple[str, Set[int]]]):
    num_items = len(data_list)
    similarity_array = np.zeros((num_items, num_items))
    extra_data = {
        "overlap": np.empty((num_items, num_items), dtype=list),
        "overlap_num": np.zeros((num_items, num_items)),
    }
    for idx_a, (group_a, dim_set_a) in enumerate(data_list):
        for idx_b, (group_b, dim_set_b) in enumerate(data_list):
            similarity_array[idx_a, idx_b] = compute_overlap_coefficient(dim_set_a, dim_set_b)
            extra_data["overlap"][idx_a, idx_b] = sorted(list(set(dim_set_a & dim_set_b)))
            extra_data["overlap_num"][idx_a, idx_b] = len(dim_set_a & dim_set_b)

    return similarity_array, extra_data


def compute_pvalues(overlap_num_matrix: np.array, p_val_dict: Dict[int, float]) -> np.array:
    return np.vectorize(lambda x: p_val_dict[int(x)])(overlap_num_matrix)


def build_statistical_significance_matrix(p_values_matrix, alpha=0.05, method="bonferroni", symmetry=False):
    num_rows = p_values_matrix.shape[0]
    num_hypotheses = int(num_rows * (num_rows + 1) / 2) - num_rows
    num_hypotheses = num_hypotheses if num_hypotheses > 0 else 999
    alpha_bonferroni = alpha / num_hypotheses

    if method == "bonferroni":
        mask = np.tril(np.ones_like(p_values_matrix, dtype=bool), k=-1)
        significance_matrix = (p_values_matrix < alpha_bonferroni) * mask
    elif method == "holm-bonferroni":
        mask = np.triu(np.ones_like(p_values_matrix)) * 9999.0
        p_values_matrix += mask
        p_values_matrix_flat = p_values_matrix.reshape(-1)
        sorting_indices = p_values_matrix_flat.argsort()
        unsorting_indices = sorting_indices.argsort()

        sorted_p_values = p_values_matrix_flat[sorting_indices][:num_hypotheses]
        alpha_holm = np.arange(1.0, num_hypotheses + 1.0)[::-1] ** -1 * alpha

        broke = False
        for k, (pval, alph) in enumerate(zip(sorted_p_values.tolist(), alpha_holm.tolist())):
            if pval > alph:
                broke = True
                break

        if not broke:
            # Needed in case we never accepted the null hypothesis
            k += 1

        # k will be equal to the first index where we do NOT reject the null hypothesis.
        # So we can accept the alternative hypothesis on all indices less than k
        # e.g., if k == 0, we always accept the null hypothesis. If k == num_hypothesis
        # we always reject the null hypothesis.
        rejected_null_sorted = [True if idx < k else False for idx in range(num_hypotheses)]

        # Pad remaining list with rejections
        rejected_null_sorted.extend([False] * (num_rows ** 2 - num_hypotheses))

        # Reverse sort
        significance_matrix = np.array(rejected_null_sorted)[unsorting_indices].reshape(num_rows, num_rows)

    if symmetry:
        # Mirror along diagonal
        significance_matrix = significance_matrix | significance_matrix.T

    return significance_matrix


def build_annotations_list(annotation_matrix):
    n = annotation_matrix.shape[0]
    annotation_list = []
    for x in range(n):
        for y in range(n):
            if not annotation_matrix[x][y]:
                continue

            if x == y:
                continue

            annotation_list.append(
                dict(
                    x=x / n, y=y / n,
                    xref='paper',
                    yref='paper',
                    text="■",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="rgb(236,136,106)")
                )
            )

    return annotation_list


"""
# Print raw overlap
display_k = 3
attributes = ["Number", "Gender and Noun Class", "Case", "Tense", "Person"]
results, mark_count = compute_overlap(results_raw, top_k)
for attr in attributes:
    counter = results[attr]
    print(f"{attr} ({mark_count[attr]}) & {', '.join([f'{k} ({v})' for k, v in counter.most_common(display_k)])} \\\\")
"""

lang_to_family = {lang: fam for lang, fam in lang_sorted_family}

def get_pvals(embedding):
# Compute p values

    num_permutations = 1000000
    k = 30
    p_vals_cache_file = f"{embedding}_{top_k}_{num_permutations}_pvals.pkl"
    if not os.path.exists(p_vals_cache_file):
        # Compute p-values for different similarities
        if embedding in ["bert-base-multilingual-cased", "xlm-roberta-base"]:
            dimensionality = 768
        elif embedding == "xlm-roberta-large":
            dimensionality = 1024
        else:
            raise Exception("Embedding has to be BERT!")
            dimensionality = 300

        reference_order = random.sample(list(range(dimensionality)), dimensionality)
        reference_top_k = reference_order[:top_k]
        similarities = []

        for i in tqdm(range(num_permutations)):
            permuted_top_k = random.sample(reference_order, top_k)
            similarities.append(compute_overlap_coefficient(set(reference_top_k), set(permuted_top_k)))
            pvals = {}

        for i in range(top_k + 1):
            observed_hypothesis = i / top_k  # What is overlap score greater than or equal to?
            permutations_match = [s for s in similarities if s >= observed_hypothesis]
            pval = len(permutations_match) / num_permutations
            print(f"P-value when sim >= {observed_hypothesis} (overlap >= {i} dims): {pval:.5f}")
            pvals[i] = pval

        with open(p_vals_cache_file, "wb") as h:
            pickle.dump(pvals, h)
    else:
        with open(p_vals_cache_file, "rb") as h:
            pvals = pickle.load(h)
    
    return pvals


### Similarity

all_sim = []
for embedding in ["bert-base-multilingual-cased", "xlm-roberta-base", "xlm-roberta-large"]:
    for attr in attributes:
        lang_sim = []
        labels, (similarity_matrix, extra_data) = compute_similarity_for_attribute(
            attr, embedding, results_raw, top_k, language_order=[x[0] for x in lang_sorted_family])

        x_labels = labels
        y_labels = [[lang_to_family[x] for x in labels], labels]
        pvals = get_pvals(embedding)

        p_values_matrix = compute_pvalues(extra_data["overlap_num"], pvals)
        annotation_matrix = build_statistical_significance_matrix(
            p_values_matrix, alpha=0.05, method="holm-bonferroni", symmetry=True)

        mask_ut=np.triu(np.ones(similarity_matrix.shape)).astype(np.bool)
        for i in range(len(similarity_matrix)):
            lang1 = x_labels[i]
            for j in range(len(similarity_matrix[i])):
                if mask_ut[i][j]:
                    continue
                else:
                    lang2 = x_labels[j]
                    overlap = similarity_matrix[i][j]
                    sim = 1 - l2v.syntactic_distance(lang1, lang2)

                    lang_sim.append([embedding, attr, lang1, lang2, overlap, sim]) 
        
        all_sim.append(lang_sim)
        lang_sim = pd.DataFrame(lang_sim, columns = ["embedding", "attribute", "lang1", "lang2", "overlap", "similarity"])
        lang_sim.to_csv(f"{attr}_overlap_similarity_{embedding}.csv")
all_sim = pd.DataFrame(all_sim, columns = ["attribute", "lang1", "lang2", "overlap", "similarity"])
all_sim.to_csv(f"overlap_similarity_{embedding}.csv")

df = []
all_emb = pd.DataFrame()
for embedding in ["bert-base-multilingual-cased", "xlm-roberta-base", "xlm-roberta-large"]:
    all_sim = pd.read_csv(f"overlap_similarity_{embedding}.csv")
    if embedding == "bert-base-multilingual-cased":
        embedding = "m-BERT"
    if embedding == "xlm-roberta-base":
        embedding = "XLM-R-base"
    if embedding == "xlm-roberta-large":
        embedding = "XLM-R-large"
    all_emb = all_emb.append(all_sim.assign(embedding = embedding),ignore_index=True)

    for attr in attributes:
        if len(all_sim[all_sim.attribute == attr]) > 1:
            cor_coef = all_sim[all_sim.attribute == attr]["overlap"].corr(all_sim[all_sim.attribute == attr]["similarity"])
            df.append([embedding, attr, cor_coef])

df = pd.DataFrame(df, columns = ["Embedding", "attribute", "correlation"]).reset_index(drop=True)
df_sim = df.pivot(index='Embedding', columns='attribute', values='correlation').reset_index().rename_axis(None, axis=1)
df_sim.set_index(['Embedding'], inplace=True)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 6.4
fig_size[1] = 6.5
plt.rcParams["figure.figsize"] = fig_size

# Use axes divider to put cbar on top
# plot heatmap without colorbar
sns.set(style="ticks", font="Times New Roman", font_scale=1.1)
ax = sns.heatmap(df_sim.round(2), annot=True, fmt="g", center=0.00, cmap="coolwarm", square=True, cbar = False)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')
# split axes of heatmap to put colorbar
ax_divider = make_axes_locatable(ax)
# define size and padding of axes for colorbar
cax = ax_divider.append_axes('bottom', size = '5%', pad = '72%')
# make colorbar for heatmap. 
# Heatmap returns an axes obj but you need to get a mappable obj (get_children)
colorbar(ax.get_children()[0], cax = cax, orientation = 'horizontal')
# locate colorbar ticks
cax.xaxis.set_ticks_position('bottom')

plt.savefig(f'./experiments/similarity_corr_all.pdf', bbox_inches='tight')
plt.figure().clear()


### Number of attribute values

# Read in list of attribute value pairs for each treebank 
# Relevant file generation script to be found in generate_list_of_probed_attribute_value_pairs.py 
treebank_attr = pd.read_csv("treebank_attribute.csv")
treebank_attr.Attribute = treebank_attr.Attribute.apply(lambda x: "POS" if x == "Part of Speech" else x)
treebank_attr.Attribute = treebank_attr.Attribute.apply(lambda x: "Gender" if x == "Gender and Noun Class" else x)

all_emb = all_emb.merge(treebank_attr[["Language","Attribute",  "No_Values"]].set_index( ['Language']), 
            how='left', left_on=['lang1', 'attribute'], right_on=['Language', 'Attribute'])
all_emb = all_emb.merge(treebank_attr[["Language", "Attribute", "No_Values"]].set_index( ['Language']), 
            how='left', left_on=['lang2', 'attribute'], right_on=['Language', 'Attribute'])
all_emb["attr_diff"] = np.abs(all_emb["No_Values_x"] - all_emb["No_Values_y"])

df_emb = []
for embedding in all_emb.embedding.unique():
    df = all_emb[all_emb.embedding == embedding]
    for attr in attributes:
        if len(df[df.attribute == attr]) > 1:
            cor_coef = df[df.attribute == attr]["overlap"].corr(df[df.attribute == attr]["attr_diff"])
            df_emb.append([embedding, attr, cor_coef])

df_emb = pd.DataFrame(df_emb, columns = ["Embedding", "attribute", "correlation"]).reset_index(drop=True)
df_emb = df_emb.pivot(index='Embedding', columns='attribute', values='correlation').reset_index().rename_axis(None, axis=1)
df_emb.set_index(['Embedding'], inplace=True)
df_emb.dropna(axis=1, how='all', inplace=True)

# Use axes divider to put cbar on top
# plot heatmap without colorbar
sns.set(style="ticks", font="Times New Roman", font_scale=1.1)
ax = sns.heatmap(df_emb.round(2), annot=True, fmt="g", center=0.00, cmap="coolwarm", square=True, cbar = False)
# split axes of heatmap to put colorbar
ax_divider = make_axes_locatable(ax)
# define size and padding of axes for colorbar
cax = ax_divider.append_axes('bottom', size = '5%', pad = '20%')
# make colorbar for heatmap. 
# Heatmap returns an axes obj but you need to get a mappable obj (get_children)
colorbar(ax.get_children()[0], cax = cax, orientation = 'horizontal')
# locate colorbar ticks
cax.xaxis.set_ticks_position('bottom')

plt.savefig(f'./experiments/no_attr_corr_all.pdf', bbox_inches='tight')