
for f in "data/ProcessedDatasets/"**/**/*; do
      echo $'\n'
      echo "$f"
      type=$(echo $f | cut -d '/' -f 2 | sed 's/synth_//'| sed 's/ /_/' )
      python src/MLM_Script.py  --file "$f"\
                                --model_arch 'bert-base-cased'\
                                --concept_vector 'data/Top10TypeVectors_10/'"$type"'_vectors.pkl'\
                                --method_label 'Top10'



      python src/MLM_Script.py  --file "$f"\
                                --model_arch 'bert-base-cased'\
                                --concept_vector 'data/Bot10TypeVectors_10/'"$type"'_vectors.pkl'\
                                --method_label 'Bot10'

      python src/MLM_Script.py  --file "$f"\
                                --model_arch 'bert-base-cased'\
                                --concept_vector 'data/RandomUniformTypeVectors_10/'"$type"'_vectors.pkl'\
                                --method_label 'Rand'


  done

done

python src/print_avg.py --res_dir "results/ProcessedDatasets" 
