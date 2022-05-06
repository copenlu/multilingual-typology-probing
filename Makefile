SHELL := /bin/bash
DATA_DIR := data/ud
UD_DIR := $(DATA_DIR)/ud-treebanks-v2.1
UD_EXTRACT_LANGUAGE := ^.*\/([a-z]+)(_[a-zA-Z]+)?-um-(dev|train|test)\.conllu$$
TREEBANKS := `cat ./scripts/languages_common.lst`
MULTILINGUAL_BERT_EMBEDDING := bert-base-multilingual-cased
XLMR_EMBEDDING := xlm-roberta-base
XLMR_LARGE_EMBEDDING := xlm-roberta-large
RESULTS_DIR := results
RESULTS_BERT_DIR := $(RESULTS_DIR)/01_bert_results
RESULTS_XLMR_DIR := $(RESULTS_DIR)/01_xlmr_results
RESULTS_XLMR_LARGE_DIR := $(RESULTS_DIR)/01_xlmr_results
GREEDY_SELECTION_SIZE := 50
GREEDY_SELECTION_CRITERION := mi

ifeq ($(shell touch .wandbtag && cat .wandbtag),)
WANDB_TAG := flexible-probing-typology
else
WANDB_TAG := $(shell cat .wandbtag)
endif

# Detect Linux vs OSX and presence of appropriate binaries
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
SED = sed
endif

ifeq ($(UNAME_S),Darwin)
ifeq (, $(shell which gsed))
$(error "GSED MISSING: You don't have gsed installed. You need to install it, e.g., 'brew install gnu-sed'.")
endif
SED = gsed
endif

# General rules
.PHONY: install
install: data/tags.yaml
	git submodule init
	git submodule update
	cp config.default.yml config.yml

data/tags.yaml:
	mkdir -p data
	cd data && wget https://raw.githubusercontent.com/unimorph/um-canonicalize/master/um_canonicalize/tags.yaml

.PHONY: clean
clean:
	rm -rf $(DATA_DIR)

# Download data
.PHONY: download
download: $(UD_DIR)

$(UD_DIR):
	mkdir -p $(DATA_DIR)
	cd $(DATA_DIR) && curl https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz > ud-treebanks-v2.6.tgz
	cd $(DATA_DIR) && tar -xzvf ud-treebanks-v2.6.tgz

## Convert UD to UM
ud_selected_dirs = $(shell for s in $(TREEBANKS); do printf "$(UD_DIR)/$$s "; done)
ud_selected_files = $(foreach d,$(ud_selected_dirs),$(wildcard $(d)/*-ud-*.conllu))
um_selected_files = $(subst -ud-,-um-,$(ud_selected_files))

output_file_template_split = $(shell for s in $(TREEBANKS); do printf "$(UD_DIR)/$$s/$$s-$(1)-$(2).pkl "; done)
output_file_template = $(call output_file_template_split,train,$(1))

.PHONY: convert
convert: $(um_selected_files)

$(um_selected_files): %:
	MATCHED_LANGUAGE="$(shell echo '$*' | $(SED) -r -e 's/$(UD_EXTRACT_LANGUAGE)/\1/')" && echo $$MATCHED_LANGUAGE && \
					 UD_FILE=`echo $(subst -um-,-ud-,$*)` && \
					 cd ud-compatibility/UD_UM && pwd && python marry.py convert --ud ../../$$UD_FILE --lang $$MATCHED_LANGUAGE

.PHONY: process

process: process_bert
process_bert: $(call output_file_template,$(MULTILINGUAL_BERT_EMBEDDING))

$(call output_file_template,$(MULTILINGUAL_BERT_EMBEDDING)): %-train-$(MULTILINGUAL_BERT_EMBEDDING).pkl:
	MATCHED_TREEBANK="$(shell echo '$*' | $(SED) -r -e 's/^.*\/(UD_[-_a-zA-Z]+)$$/\1/')" && echo $$MATCHED_TREEBANK && \
					 python preprocess_treebank.py $$MATCHED_TREEBANK --bert $(MULTILINGUAL_BERT_EMBEDDING)

## Experiments
valid_languages = $(shell while IFS=$$'\n' read -r line; do echo $$line; done<./scripts/languages_common.lst)
valid_properties = $(shell while IFS=$$'\n' read -r line; do echo $$line; done<./scripts/properties.lst)

bert_results_file_name = $(RESULTS_BERT_DIR)/$(1)---$(2)---$(MULTILINGUAL_BERT_EMBEDDING).json
#bert_results_files = $(shell for L in $(valid_languages); do for P in $(valid_properties); do echo $(call bert_results_file_name,$$L,$$P); done done)
bert_results_files = $(foreach L,$(valid_languages),$(foreach P,$(valid_properties),$(call bert_results_file_name,$(L),$(P))))
bert_results_files_for_language = $(foreach P,$(valid_properties),$(call bert_results_file_name,$(1),$(P)))

bert_results_file_name_manual = $(RESULTS_BERT_DIR)/manual-$(1)---$(2)---$(MULTILINGUAL_BERT_EMBEDDING).json
#bert_results_files = $(shell for L in $(valid_languages); do for P in $(valid_properties); do echo $(call bert_results_file_name,$$L,$$P); done done)
bert_results_files_manual = $(foreach L,$(valid_languages),$(foreach P,$(valid_properties),$(call bert_results_file_name_manual,$(L),$(P))))
bert_results_files_for_language_manual = $(foreach P,$(valid_properties),$(call bert_results_file_name_manual,$(1),$(P)))

xlmr_results_file_name = $(RESULTS_XLMR_DIR)/$(1)---$(2)---$(XLMR_EMBEDDING).json
xlmr_results_files = $(foreach L,$(valid_languages),$(foreach P,$(valid_properties),$(call xlmr_results_file_name,$(L),$(P))))
xlmr_results_files_for_language = $(foreach P,$(valid_properties),$(call xlmr_results_file_name,$(1),$(P)))

xlmr_large_results_file_name = $(RESULTS_XLMR_LARGE_DIR)/$(1)---$(2)---$(XLMR_LARGE_EMBEDDING).json
xlmr_large_results_files = $(foreach L,$(valid_languages),$(foreach P,$(valid_properties),$(call xlmr_large_results_file_name,$(L),$(P))))
xlmr_large_results_files_for_language = $(foreach P,$(valid_properties),$(call xlmr_large_results_file_name,$(1),$(P)))

$(bert_results_files): $(RESULTS_BERT_DIR)/%---$(MULTILINGUAL_BERT_EMBEDDING).json:
	TB=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\1/') && \
	   ATTR=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\2/') && \
	   python -u run.py --language $$TB --attribute $$ATTR --trainer poisson --output-file $@ --wandb --wandb-tag $(WANDB_TAG) \
		   greedy --selection-size $(GREEDY_SELECTION_SIZE) --selection-criterion $(GREEDY_SELECTION_CRITERION)

01_bert_ALL: $(bert_results_files)

# Constructs rules of the form "01_bert_UD_English-EWT" for quick running
define 01_bert_RULE
01_bert_$(1): $(call bert_results_files_for_language,$(1))
endef
$(foreach lang,$(valid_languages),$(eval $(call 01_bert_RULE,$(lang))))


###

$(bert_results_files_manual): $(RESULTS_BERT_DIR)/%---$(MULTILINGUAL_BERT_EMBEDDING).json:
	TB=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\1/') && \
	   ATTR=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\2/') && \
	   python -u run.py --language $$TB --attribute $$ATTR --trainer poisson --output-file $@ --wandb --wandb-tag $(WANDB_TAG) \
		   --l1-weight 1e-5 --l2-weight 1e-5 file --file experiments/01_dimension-tests.json

02_bert_ALL: $(bert_results_files_manual)

# Constructs rules of the form "01_bert_UD_English-EWT" for quick running
define 02_bert_RULE
02_bert_$(1): $(call bert_results_files_for_language_manual,$(1))
endef
$(foreach lang,$(valid_languages),$(eval $(call 02_bert_RULE,$(lang))))

###

$(xlmr_results_files): $(RESULTS_XLMR_DIR)/%---$(XLMR_EMBEDDING).json:
	TB=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\1/') && \
	   ATTR=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\2/') && \
	   python -u run.py --language $$TB --attribute $$ATTR --trainer poisson --embedding xlm-roberta-base --output-file $@ --wandb --wandb-tag $(WANDB_TAG) \
		   greedy --selection-size $(GREEDY_SELECTION_SIZE) --selection-criterion $(GREEDY_SELECTION_CRITERION)


01_xlmr_base_ALL: $(xlmr_results_files)

# Constructs rules of the form "01_bert_UD_English-EWT" for quick running
define 01_xlmr_RULE
01_xlmr_$(1): $(call xlmr_results_files_for_language,$(1))
endef
$(foreach lang,$(valid_languages),$(eval $(call 01_xlmr_RULE,$(lang))))



$(xlmr_large_results_files): $(RESULTS_XLMR_LARGE_DIR)/%---$(XLMR_LARGE_EMBEDDING).json:
	TB=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\1/') && \
	   ATTR=$(shell echo $* | $(SED) -re 's/^(.*)---(.*)$$/\2/') && \
	   python -u run.py --language $$TB --attribute $$ATTR --trainer poisson --embedding xlm-roberta-large --output-file $@ --wandb --wandb-tag $(WANDB_TAG) \
		   greedy --selection-size $(GREEDY_SELECTION_SIZE) --selection-criterion $(GREEDY_SELECTION_CRITERION)


01_xlmr_large_ALL: $(xlmr_large_results_files)

# Constructs rules of the form "01_bert_UD_English-EWT" for quick running
define 01_xlmr_large_RULE
01_xlmr_large_$(1): $(call xlmr_large_results_files_for_language,$(1))
endef
$(foreach lang,$(valid_languages),$(eval $(call 01_xlmr_large_RULE,$(lang))))


