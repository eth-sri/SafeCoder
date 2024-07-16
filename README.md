# SafeCoder: Instruction Tuning for Secure Code Generation <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
SafeCoder enables large language models (LLMs) to learn to generate secure code during instruction tuning. This is the official repository for [our ICML 2024 paper](https://arxiv.org/abs/2402.09497).

## Setup
First, install Python dependencies:
```console
pip install -r requirements.txt
pip install -e .
```
Then, install [GitHub CodeQL](https://codeql.github.com/), which will be used for evaluating the security of LLM-generated code:
```console
./setup_codeql.sh
```
Finally, set up different programming languages studied in this work (`sudo` rights required):
```console
./setup_langs.sh
```

## Training
Run the following command to fine-tune an pretrained LLM with SafeCoder:
```console
python train.py --pretrain_name starcoderbase-1b --output_name starcoderbase-1b-safecoder --datasets evol sec-desc sec-new-desc
```
Here, `--pretrain_name` specifies the base pretrained LLM, `--output_name` denotes the user-provided name of the fine-tuned model, and `--datasets` represents a list of datasets used for training (see [the datasets section](#datasets) for more details). We also provide fine-tuned versions of Mistral-7B ([link](https://files.sri.inf.ethz.ch/safecoder/mistral-7b-lora-safecoder.tar.gz)) and CodeLlama-7B ([link](https://files.sri.inf.ethz.ch/safecoder/codellama-7b-lora-safecoder.tar.gz)), such that the user does not necessarily need to perform fine-tuning by themselves.

## Evaluation
Our evaluation covers various benchmarks concerning security and utility. To evaluate the security of generated code, run the following commands:
```console
python sec_eval.py --output_name starcoderbase-1b-safecoder --model_name starcoderbase-1b-safecoder --eval_type trained
python sec_eval.py --output_name starcoderbase-1b-safecoder --model_name starcoderbase-1b-safecoder --eval_type trained-new
python print_results.py --eval_name starcoderbase-1b-safecoder --eval_type trained-joint --detail
```

For utility, we consider the following benchmarks:
```console
# HumanEval, with temperature 0.2
./func_eval.sh human_eval starcoderbase-1b-safecoder-0.2 starcoderbase-1b-safecoder 0.2
python print_results.py --eval_name starcoderbase-1b-safecoder-0.2 --eval_type human_eval

# MBPP, with temperature 0.2
./func_eval.sh mbpp starcoderbase-1b-safecoder-0.2 starcoderbase-1b-safecoder 0.2
python print_results.py --eval_name starcoderbase-1b-safecoder-0.2 --eval_type mbpp

# MMLU
python mmlu_eval.py --output_name starcoderbase-1b-safecoder --model_name starcoderbase-1b-safecoder
python print_results.py --eval_name starcoderbase-1b-safecoder --eval_type mmlu

# TruthfulQA
python truthfulqa_eval.py --output_name starcoderbase-1b-safecoder --model_name starcoderbase-1b-safecoder
python print_results.py --eval_name starcoderbase-1b-safecoder --eval_type tqa
```

## Datasets
The repository contains two utility datasets, [`evol`](data_train_val/train/evol.jsonl) and [`lmsys`](data_train_val/val/lmsys.jsonl). In the paper, `evol` is used with code-specific LLMs and `lmsys` is used with general-purpose LLMs. We also have two security datasets, [`sec-desc`](data_train_val/train/sec-desc.jsonl) and [`sec-new-desc`](data_train_val/val/sec-new-desc.jsonl). `sec-desc` is adapted from our previous work [`SVEN`](https://github.com/eth-sri/sven), while `sec-new-desc` is constructed within this work (see [Section 5 in our paper](https://arxiv.org/pdf/2402.09497) for more details). [`trained`](data_eval/sec_eval/trained/) and [`trained-new`](data_eval/sec_eval/trained-new/) correspond to the evaluation datasets for `sec-desc` and `sec-new-desc`, respectively.

## Citation
```
@inproceedings{safecoder,
  author       = {Jingxuan He, Mark Vero, Gabriela Krasnopolska, and Martin Vechev},
  title        = {Instruction Tuning for Secure Code Generation},
  booktitle    = {ICML},
  year         = {2024},
  url          = {https://arxiv.org/abs/2402.09497},
}
```