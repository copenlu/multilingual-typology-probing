from typing import List, Optional, Dict, Callable, Any, Set
import yaml
from word import Word
import torch


AttributeToValuesDict = Dict[str, List[str]]
AttributeValueCounter = Dict[str, Dict[str, int]]


def load_dimensions_to_features(tags_path: str) -> Dict[str, List[str]]:
    with open(tags_path, 'r') as h:
        unimorph_tags = yaml.full_load(h)

    return unimorph_tags["categories"]


DIMENSIONS_TO_FEATURES = load_dimensions_to_features("utils/tags.yaml")
FEATURES_TO_DIMENSIONS = {feat: k for k, v in DIMENSIONS_TO_FEATURES.items()
                          for feat in v}


class Reader:
    def __init__(self, words: List[Word], attribute_to_values_dict: AttributeToValuesDict):
        """
        A standard reader. This class should not be instantiated directly. Instead,
        use UnimorphReader or UDTreebankReader.

        The reader constructor takes a word list and a dict of attributes and all the values they can take.
        It will ensure that examples for attribute-values that aren't in the dict are discarded.
        """
        self._unimorph_attributes_to_values_dict: AttributeToValuesDict = attribute_to_values_dict

        # Discard invalid attribute-values
        modified_words: List[Word] = []
        for w in words:
            modified_attr_vals: Dict[str, str] = {}
            for attr in w.get_attributes():
                if attr not in self._unimorph_attributes_to_values_dict:
                    # Untracked attribute, so skip it
                    continue

                val = w.get_attribute(attr)
                if val not in self._unimorph_attributes_to_values_dict[attr]:
                    # Untracked value, so skip it
                    continue

                modified_attr_vals[attr] = val

            # Create modified word
            modified_words.append(Word(w.get_word(), w.get_embedding(), w.get_count(), modified_attr_vals))

        self._words: List[Word] = modified_words

        self._cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_attribute_value_counter(cls, raw_words: List[Word]) -> AttributeValueCounter:
        """
        Given a list of words, this returns a dictionary containing the counts of every attribute-value
        in the list of words.
        """
        counter: Dict[str, Dict[str, int]] = {}
        tracker_unique: Dict[str, Dict[str, Set]] = {}
        for item in raw_words:
            for attr in item.get_attributes():
                val = item.get_attribute(attr)

                # Check if this form has already appeared
                if attr not in tracker_unique:
                    tracker_unique[attr] = {}

                if val not in tracker_unique[attr]:
                    tracker_unique[attr][val] = set()

                if item.get_word() in tracker_unique[attr][val]:
                    # This form has already appeared, so we skip it from the count
                    continue

                # Novel word form--add it to unique tracker
                tracker_unique[attr][val].add(item.get_word())

                # Add missing attributes
                if attr not in counter:
                    counter[attr] = {}

                # Add discovered values
                if val not in counter[attr]:
                    counter[attr][val] = 0

                counter[attr][val] += 1

        return counter

    @classmethod
    def get_attributes_to_values_dict_from_counters(
            self, counters: List[AttributeValueCounter], min_count: int = 50) -> AttributeToValuesDict:
        """
        Given a list of attribute-value counters, returns an attribute-value dict that ensures that:

            i) Every attribute-value has at least `min_count` examples in _every_ counter. This ensures
               that the training and test set have enough instances to compute accuracies and model parameters.
            ii) Ever attribute has at least two valid values.

        This attribute-value dict is subsequently applies to filter all entries from our readers.
        """
        # Pick first counter as starting point for our dict.
        attr_val_dict = {a: list(counters[0][a].keys()) for a in counters[0].keys()}

        # Using the entries in the first counter, iterate through _all_ counters and remove entries
        # where: (i) attr does not appear on all counters,
        # (ii) attr-val combination does not exist in counter, and (iii) attr-val count < min_count
        for counter in counters:
            for a, vs in list(attr_val_dict.items()):
                # Remove whole attribute if it is not in the counter
                if a not in counter:
                    del attr_val_dict[a]
                    continue

                for v in list(vs):
                    # Remove if value is not in counter
                    if v not in counter[a]:
                        attr_val_dict[a].remove(v)
                        continue

                    # Remove is min_count not achieved by counter
                    if counter[a][v] < min_count:
                        attr_val_dict[a].remove(v)

        # Remove any attributes that have less than two values
        for a, vs in list(attr_val_dict.items()):
            if len(attr_val_dict[a]) < 2:
                del attr_val_dict[a]

        return attr_val_dict

    def get_dimensionality(self) -> int:
        return self._words[0].get_embedding().shape[0]

    def get_words(self) -> List[Word]:
        return self._words

    def get_words_with_filter_from_cache(
            self, cache_key: str, filter: Optional[Callable] = None) -> List[Word]:
        """
        Given some cache_key chosen to correspond to the filter being used (e.g. "attribute_Tense"),
        returns the word list after applying the filter, using the cache if possible.
        """
        if cache_key not in self._cache:
            self._cache[cache_key] = {}

        if "words" not in self._cache[cache_key]:
            if filter is not None:
                self._cache[cache_key]["words"] = [w for w in self.get_words() if filter(w)]
            else:
                self._cache[cache_key]["words"] = [w for w in self.get_words()]

        return self._cache[cache_key]["words"]

    def get_embeddings_with_filter_from_cache(
            self, cache_key: str, filter: Optional[Callable] = None) -> torch.Tensor:
        """
        Given some cache_key chosen to correspond to the filter being used (e.g. "attribute_Tense"),
        returns the list of embeddings after applying the filter, using the cache if possible.
        """
        if cache_key not in self._cache:
            self._cache[cache_key] = {}

        if "embeddings" not in self._cache[cache_key]:
            if filter is not None:
                self._cache[cache_key]["embeddings"] = torch.tensor(
                    [w.get_embedding() for w in self.get_words() if filter(w)])
            else:
                self._cache[cache_key]["embeddings"] = torch.tensor(
                    [w.get_embedding() for w in self.get_words()])

        return self._cache[cache_key]["embeddings"]

    def get_values_with_filter_from_cache(
            self, attribute: str, cache_key: str, value_model,
            filter: Optional[Callable] = None) -> torch.Tensor:
        if cache_key not in self._cache:
            self._cache[cache_key] = {}

        key = "{}_value".format(attribute)
        if key not in self._cache[cache_key]:
            if filter is not None:
                self._cache[cache_key][key] = value_model.get_value_ids(
                    [w.get_attribute(attribute) for w in self.get_words() if filter(w)])
            else:
                self._cache[cache_key][key] = value_model.get_value_ids(
                    [w.get_attribute(attribute) for w in self.get_words()])

        return self._cache[cache_key][key]

    def has_attribute(self, attribute: str) -> bool:
        if attribute in self._get_implemented_attributes_to_values_dict().keys():
            return True

        return False

    def _get_implemented_attributes_to_values_dict(self) -> Dict[str, List[str]]:
        attr_to_val = self._get_unimorph_attributes_to_values_dict()
        return dict(attr_to_val, **self._get_custom_attributes_to_values_dict())

    def _get_unimorph_attributes_to_values_dict(self) -> Dict[str, List[str]]:
        return self._unimorph_attributes_to_values_dict

    def get_valid_attributes(self) -> List[str]:
        return list(self._get_implemented_attributes_to_values_dict().keys())

    def get_valid_attribute_values(self, attribute: str) -> List[str]:
        if not self.has_attribute(attribute):
            raise Exception("Invalid attribute.")

        return self._get_implemented_attributes_to_values_dict()[attribute]

    def get_attributes_from_features(self, unimorph_features: List[str]) -> Dict[str, str]:
        """
        Given a list of Unimorph features associated with a word, returns a dict
        that associates each feature with a tracked attribute.
        """
        res = AttributeDict()

        # Auto-add specified unimorph attributes
        res.build_from_attribute_value_dict(
            self._get_implemented_attributes_to_values_dict(), unimorph_features)

        # Add language-specific attributes
        res.update(self._get_language_specific_attributes(unimorph_features))
        return res

    @classmethod
    def read(self, path: List[str]) -> List[Word]:
        """
        Should be overriden with the logic to read all words in the dataset and (ii) discover
        the values each unimorph attribute can take.
        """
        raise NotImplementedError

    def _get_custom_attributes_to_values_dict(self) -> Dict[str, List[str]]:
        """
        Can be overriden to return a dictionary of custom attributes, with a list
        of the values they can take on.
        """
        return {}

    def _get_language_specific_attributes(self, unimorph_features: List[str]) -> Dict[str, str]:
        """
        If we are creating custom features using some other information source, or my
        merging/composing UniMorph features, we create and return them by overriding this
        function in subclasses.

        We must also override self._get_custom_attributes_to_values_dict so that the values that
        the custom attributes can take are made available there.
        """
        return {}


class AttributeDict(Dict[str, str]):
    def accept_first_match(self, matches: List[str], unimorph_features: List[str]) -> Optional[str]:
        for uf in unimorph_features:
            if uf in matches:
                return uf

        return None

    def add_if_match(self, attribute: str, matches: List[str],
                     unimorph_features: List[str]) -> Dict[str, str]:
        first_match = self.accept_first_match(matches, unimorph_features)
        if first_match is not None:
            self[attribute] = first_match

        return self

    def build_from_attribute_value_dict(self, attribute_value_dict: Dict[str, List[str]],
                                        unimorph_features: List[str]) -> Dict[str, str]:
        for attr, values in attribute_value_dict.items():
            self.add_if_match(attr, values, unimorph_features)

        return self
