
for f in "data/ProcessedDatasets/"**/**/*; do
  echo $'\n'
  echo "$f"
  type=$(echo $f | cut -d '/' -f 2 | sed 's/synth_//'| sed 's/ /_/' )
  echo $k


  python src/MLM_Script.py  --file "$f"\
                            --model_arch 'bert-base-cased'\
                            --concept_vector 'data/TypeVectors_10/'"$type"'_vectors.pkl'\
                            --k 0\
                            --manual_k\
                            --method_label 'B'


  python src/MLM_Script.py  --file "$f"\
                            --model_arch 'bert-base-cased'\
                            --concept_vector 'data/TypeVectors_10/'"$type"'_vectors.pkl'\
                            --token_baseline\
                            --method_label 'BTo'
   
  python src/MLM_Script_Baseline.py   --file "$f"\
                                      --model_arch 'bert-base-cased'\
                                      --concept_vector 'data/TypeVectors_10/'"$type"'_vectors.pkl'\
                                      --method_label 'PostTE'


  python src/MLM_Script.py  --file "$f"\
                            --model_arch 'bert-base-cased'\
                            --concept_vector 'data/TypeVectors_10/'"$type"'_vectors.pkl'\
                            --method_label 'BTe'

done


