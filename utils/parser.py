from typing import Dict, List
import yaml


tags_file = "./utils/tags.yaml"
with open(tags_file, 'r') as h:
    _UNIMORPH_ATTRIBUTE_VALUES = yaml.full_load(h)["categories"]

_UNIMORPH_VALUES_ATTRIBUTE = {v: k for k, vs in _UNIMORPH_ATTRIBUTE_VALUES.items() for v in vs}


def parse_unimorph_features(features: List[str]) -> Dict[str, str]:
    final_attrs: Dict[str, str] = {}
    for x in features:
        if "/" in x:
            # NOTE: Can we handle disjunctions in a better way?
            continue
        elif x == "{CMPR}":
            # I am assuming they meant to type "CMPR" and not "{CMPR}"
            final_attrs["Comparison"] = "CMPR"
        elif x == "PST+PRF":
            # The past perfect is a common feature of Latin, Romanian, and Turkish annotations.
            # I assign it to Tense due aspect having already been assigned to something different in Turkish,
            # and since "PST" comes first.
            final_attrs["Tense"] = x
        elif x.startswith("ARG"):
            # Argument marking (e.g. in Basque) is labelled with ARGX where X is the actual feature.
            v = x[3:]
            final_attrs[_UNIMORPH_VALUES_ATTRIBUTE[v]] = v
        elif x == "NDEF":
            # I believe NDEF is used to designate Hebrew's indefinite nouns
            final_attrs["Definiteness"] = "INDF"
        elif "+" in x:
            # We handle conjunctive statements by creating a new value for them.
            # We canonicalize the feature by sorting the composing conjuncts alphabetically.
            values = x.split("+")
            attr = _UNIMORPH_VALUES_ATTRIBUTE[values[0]]
            for v in values:
                if attr != _UNIMORPH_VALUES_ATTRIBUTE[v]:
                    raise Exception("Conjunctive values don't all belong to the same dimension.")

            final_attrs[attr] = "+".join(sorted(values))
        elif "PSS" in x:
            final_attrs["Possession"] = x
        elif "LGSPEC" in x:
            # We discard language-specific features as this is not a canonical unimorph dimension
            continue
        else:
            if x not in _UNIMORPH_VALUES_ATTRIBUTE:
                continue

            final_attrs[_UNIMORPH_VALUES_ATTRIBUTE[x]] = x

    return final_attrs
