# Under Construction
# You Are My Type! Type Embeddings for Pre-trained Language Models @ EMNLP 2022 (Findings)
[(Paper)](https://www.eurecom.fr/publication/7095/download/data-publi-7095.pdf) 

One reason for the positive impact of Pretrained Language Models (PLMs) in NLP tasks is their ability to encode semantic types, such
as ‘European City’ or ‘Woman’. While previous work has analyzed such information in the context of interpretability, it is not clear how to use types to steer the PLM output. For example, in a cloze statement, it is desirable to steer the model to generate a token that satisfies a user-specified type, e.g., predict a date rather than a location. 

In this work, we introduce Type Embeddings (TEs), an input embedding that promotes desired types in a PLM. Our proposal is to define a type by a small set of word examples. We empirically study the ability of TEs both in representing types and in steering masking  redictions without changes to the prompt text in BERT. Finally, using the LAMA datasets, we show how TEs highly improve the precision in extracting facts from PLMs.

This repo contains the required code for running the experiments of the associated paper.

## Instalation

## Run Experiments


## Contact Us
For any inquiries, feel free to [contact us](mailto:saeedm@eurecom.fr), or raise an issue on Github.


## Reference
You can cite our work:

```

```

