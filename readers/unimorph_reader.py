from typing import List
from tqdm import tqdm
from readers.base import Reader
from utils.word import Word
from utils.parser import parse_unimorph_features

from embedders.embedders import Embedder


class UnimorphReader(Reader):
    @classmethod
    def read(cls, paths: List[str], embedder: Embedder) -> List[Word]:
        # Actually process file
        raw_words = []
        for path in paths:
            with open(path, "r") as h:
                for c, line in tqdm(enumerate(h)):
                    if line == "\n":
                        # Skip blanks
                        continue

                    line = line.strip()
                    chunks = line.split('\t')
                    stem, word, features = chunks[0], chunks[1], chunks[2]  # noqa
                    feature_list = features.split(';')

                    raw_words.append((word, feature_list))

        words = []
        for word, feature_list in tqdm(raw_words):
            # We auto-create some unimorph features and
            # create some proxy features that are not available in UniMorph
            # (e.g. "3sg") with this function call.
            attrs = parse_unimorph_features(feature_list)

            words.append(Word(word, embedder.get_embedding(word), 1, attrs))

        return words
        """
        # Keep only part of the data (either train/test split)
        random.seed(0)
        random.shuffle(raw_words)
        raw_words_len = len(raw_words)
        if self._test_split:
            raw_words = raw_words[int(self._train_split_size * raw_words_len):]
        else:
            raw_words = raw_words[:int(self._train_split_size * raw_words_len)]
        """

        # Automatically detect unimorph features
        # self._unimorph_attributes_to_values_dict = self.discover_attribute_values(
        #     raw_words, self._get_unimorph_attributes())


"""
class EnglishUnimorphReader(UnimorphReader):
    def __init__(self, path: str, embedder: Embedder, limit: Optional[int] = None,
                 test_split: bool = False) -> None:
        super().__init__(path, embedder, limit, test_split)

    @overrides
    def _get_custom_attributes_to_values_dict(self) -> Dict[str, List[str]]:
        return {
            # A random attribute---MI should be (close to) 0.0 for all dimensions
            "Random": ["True", "False"],
            "3sg": ["3sg", "Non-3sg"],
        }

    @overrides
    def _get_language_specific_attributes(self, unimorph_features: List[str]) -> Dict[str, str]:
        attrs = {}

        three_sg = self.get_attribute_3sg(unimorph_features)
        if three_sg:
            attrs["3sg"] = three_sg

        random = self.get_attribute_random(unimorph_features)
        if random:
            attrs["Random"] = random

        return attrs

    def get_attribute_3sg(self, unimorph_features: List[str]) -> Optional[str]:
        if "3" in unimorph_features and "SG" in unimorph_features:
            return "3sg"

        if "NFIN" in unimorph_features:
            return "Non-3sg"

        return None

    def get_attribute_random(self, unimorph_features: List[str]) -> Optional[str]:
        if random.random() > 0.5:
            return "True"
        else:
            return "False"


class PortugueseUnimorphReader(UnimorphReader):
    def __init__(self, path: str, embedder: Embedder, limit: Optional[int] = None,
                 test_split: bool = False) -> None:
        super().__init__(path, embedder, limit, test_split)

    @overrides
    def _get_unimorph_attributes(self) -> List[str]:
        return ["Tense", "Aspect", "Mood", "Number", "Person", "Part of Speech", "Gender and Noun Class"]

    @overrides
    def _get_custom_attributes_to_values_dict(self) -> Dict[str, List[str]]:
        return {}

    @overrides
    def _get_language_specific_attributes(self, unimorph_features: List[str]) -> Dict[str, str]:
        return {}
"""
