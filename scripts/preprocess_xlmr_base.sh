for CORPUS in $(cat scripts/languages_common.lst); do
  echo "python preprocess_treebank.py $CORPUS --xlmr xlm-roberta-base" --use-gpu
  python preprocess_treebank.py $CORPUS --xlmr xlm-roberta-base --use-gpu
done
