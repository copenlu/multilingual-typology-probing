CONVERSION_SCRIPT_DIR=../ud-compatibility/UD_UM/
UD_FOLDER=../../flexible-probing-typology/data/ud/ud-treebanks-v2.1/

UD_LANGUAGE_PATTERN="\/([a-z]*)(_[a-zA-Z_]*)*-ud-[a-z\-]*\.conllu"

orig_dir=$(pwd)
cd $CONVERSION_SCRIPT_DIR

for f in $(find $UD_FOLDER -maxdepth 2 -type f | grep .conllu); do
  if [[ $f =~ $UD_LANGUAGE_PATTERN ]]; then
    lang_code=${BASH_REMATCH[1]}
    echo "Converting $f using langauge $lang_code"
    python marry.py convert --ud $f -l $lang_code
  fi
done

cd $orig_dir
