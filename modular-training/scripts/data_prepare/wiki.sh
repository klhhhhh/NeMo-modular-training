python /global/homes/k/klhhhhh/NeMo-modular-training/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
--input=/pscratch/sd/k/klhhhhh/text/wiki_all.json \
--json-keys=text \
--tokenizer-library=megatron \
--vocab /pscratch/sd/k/klhhhhh/dataset/gpt2-datasets/gpt2-vocab.json \
--dataset-impl mmap \
--tokenizer-type GPT2BPETokenizer \
--merge-file /pscratch/sd/k/klhhhhh/dataset/gpt2-datasets/gpt2-merges.txt \
--output-prefix=/pscratch/sd/k/klhhhhh/dataset/nemo/wiki/hfbpe_gpt_training_data \
--append-eod \
--workers=32