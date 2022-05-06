for CORPUS in $(cat scripts/languages_common.lst); do
  echo "python preprocess_treebank.py $CORPUS --xlmr xlm-roberta-large" --use-gpu
  python preprocess_treebank.py $CORPUS --xlmr xlm-roberta-large --use-gpu
done
