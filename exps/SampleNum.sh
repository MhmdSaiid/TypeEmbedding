
for f in "data/ProcessedDatasets/"**/**/*; do
  for n in 5 10 15 20 50; do
      echo $'\n'
      echo "$f"
      type=$(echo $f | cut -d '/' -f 2 | sed 's/synth_//'| sed 's/ /_/' )
      echo $n
      python src/MLM_Script.py  --file "$f"\
                                --model_arch 'bert-base-cased'\
                                --concept_vector 'data/TypeVectors_'"$n"'/'"$type"'_vectors.pkl'
  done

done


