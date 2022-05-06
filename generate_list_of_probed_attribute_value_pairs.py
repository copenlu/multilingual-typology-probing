from readers.ud_treebank_reader import UDTreebankReader
import pycountry
import pandas as pd

from argparse import ArgumentParser


# Select one of the embeddings
embedding = "bert-base-multilingual-cased"

languages = []
with open("./scripts/languages_common_coded.lst", "r") as h:
    for l in h:
        treebank, language_code = l.split()
        language = pycountry.languages.get(alpha_2=language_code).alpha_3
        languages.append((treebank, language))

    languages = sorted(languages)

    treebank_attr = []
    for (treebank_name, language) in languages:
        try:
            # Create Reader for English UD Treebank
            treebank = UDTreebankReader.get_treebank_file(treebank_name, embedding)
            treebank_valid = UDTreebankReader.get_treebank_file(treebank_name, embedding, valid_file=True)
            treebank_test = UDTreebankReader.get_treebank_file(treebank_name, embedding, test_file=True)
            
            words = UDTreebankReader.read([treebank])
            words_valid = UDTreebankReader.read([treebank_valid])
            words_test = UDTreebankReader.read([treebank_test])

            counters = [
                UDTreebankReader.get_attribute_value_counter(words),
                UDTreebankReader.get_attribute_value_counter(words_valid),
                UDTreebankReader.get_attribute_value_counter(words_test)
            ]
            attr_vals_dict = UDTreebankReader.get_attributes_to_values_dict_from_counters(counters, min_count=0)

            for attr, values in attr_vals_dict.items():
                treebank_attr.append([language, attr, len(values)])
        except:
            continue
    
treebank_attr = pd.DataFrame(treebank_attr, columns = ["Language", "Attribute", "No_Values"])  
treebank_attr.to_csv("treebank_attribute.csv")