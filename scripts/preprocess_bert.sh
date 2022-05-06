for CORPUS in $(cat scripts/languages_common.lst); do
  echo "python preprocess_treebank.py $CORPUS --bert bert-base-multilingual-cased" --use-gpu
  python preprocess_treebank.py $CORPUS --bert bert-base-multilingual-cased --use-gpu
done
