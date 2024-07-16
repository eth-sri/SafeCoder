import os
import ast
import sys
import torch
import random
import difflib
import logging
import tempfile
import warnings
import sacrebleu
import subprocess
import numpy as np
from tabulate import tabulate
from termcolor import colored
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .constants import PRETRAINED_MODELS, CHAT_MODELS
from .sven_models import GPTBigCodeForPrefix, PhiPrefix

logger = logging.getLogger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_logging(args, log_file):
    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )
    args.logger = logger

def visualize_pair(src_before, src_after, tokenizer):
    be_before = tokenizer.encode_plus(src_before)
    tokens_before = be_before.data['input_ids']
    tokens_before_str = list(map(lambda i: str(i), tokens_before))

    be_after = tokenizer.encode_plus(src_after)
    tokens_after = be_after.data['input_ids']
    tokens_after_str = list(map(lambda i: str(i), tokens_after))

    diffs = difflib.ndiff(tokens_before_str, tokens_after_str, linejunk=None, charjunk=None)
    for d in diffs:
        if d.startswith('- '):
            print(colored(tokenizer.decode(int(d[2:])), 'red', attrs=['reverse']), end='')
        elif d.startswith('+ '):
            print(colored(tokenizer.decode(int(d[2:])), 'green', attrs=['reverse']), end='')
        elif d.startswith('  '):
            print(tokenizer.decode(int(d[2:])), end='')
        elif d.startswith('? '):
            pass
        else:
            assert False
    print()

def visualize_weights(tokens, weights, tokenizer, color='green'):
    for t, w in zip(tokens, weights):
        s = tokenizer.decode([t])
        if w:
            print(colored(s, color, attrs=['reverse']), end='')
        else:
            print(s, end='')
    print()

def load_model(model_name, args):
    """
    Important note:
    This load function will only work for lora models if they are saved in the following pattern:
        <pretrained_base_model_name>-lora<whatever_else>
    """
    if '-lora' in model_name:

        pretrained_name = model_name.split('-lora')[0]
        pretrained_model_dir = PRETRAINED_MODELS[pretrained_name]
        if 'checkpoint-epoch' in model_name:
            fine_tuned_model_dir = os.path.join(args.model_dir, model_name)
        else:
            fine_tuned_model_dir = os.path.join(args.model_dir, model_name, 'checkpoint-last')
        assert os.path.exists(fine_tuned_model_dir)

        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, device_map='auto', trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, fine_tuned_model_dir)
        model = model.merge_and_unload()

    elif '-sven' in model_name: # happens during testing

        pretrained_name = model_name.split('-sven')[0]
        pretrained_model_dir = os.path.join(args.model_dir, pretrained_name, 'checkpoint-last')

        if 'starcoderbase' in pretrained_name:
            model_class = GPTBigCodeForPrefix
        elif 'phi-2' in pretrained_name:
            model_class = PhiPrefix
        else:
            raise NotImplementedError()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
        model = model_class.from_pretrained(pretrained_model_dir, device_map='auto', vocab_size=len(tokenizer))

        if 'checkpoint-epoch' in model_name:
            model_dir = os.path.join(args.model_dir, model_name)
        else:
            model_dir = os.path.join(args.model_dir, model_name, 'checkpoint-last')
        assert os.path.exists(model_dir)
        prefix_file = os.path.join(model_dir, 'pytorch_model.bin')
        model.prefix_params.load_state_dict(torch.load(prefix_file))

    elif hasattr(args, 'sven') and args.sven: # happens during training

        pretrained_name = model_name
        pretrained_model_dir = os.path.join(args.model_dir, model_name, 'checkpoint-last')

        if 'starcoderbase' in pretrained_name:
            model_class = GPTBigCodeForPrefix
        elif 'phi-2' in pretrained_name:
            model_class = PhiPrefix
        else:
            raise NotImplementedError()
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
        model = model_class.from_pretrained(pretrained_model_dir, device_map='auto', vocab_size=len(tokenizer))

        for n, p in model.named_parameters():
            if n.startswith('prefix_params'):
                p.requires_grad = True
            else:
                p.requires_grad = False
        with torch.no_grad():
            for param in model.prefix_params:
                param.fill_(0.0)

    else:

        if model_name in PRETRAINED_MODELS:
            model_dir = PRETRAINED_MODELS[model_name]
        elif model_name in CHAT_MODELS:
            model_dir = CHAT_MODELS[model_name]
        else:
            if 'checkpoint-epoch' in model_name:
                model_dir = os.path.join(args.model_dir, model_name)
            else:
                model_dir = os.path.join(args.model_dir, model_name, 'checkpoint-last')
            assert os.path.exists(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if model_name in PRETRAINED_MODELS or model_name == 'deepseek':
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', trust_remote_code=True)
        else:    
            model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', trust_remote_code=True, **{'vocab_size': len(tokenizer)})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def get_cp_args(info):
    if info['class_path'] == '':
        cp_args = ''
    else:
        paths = info['class_path'].split(':')
        paths = list(map(lambda p: os.path.realpath(p), paths))
        cp_args = '-cp {}'.format(':'.join(paths))
    return cp_args

def try_parse(code, info):
    lang = info['language']
    if lang == 'py':
        try:
            ast.parse(code)
            return 0
        except:
            return 1
    elif lang in ('c', 'js', 'rb', 'jsx'):
        if lang == 'c':
            cmd = 'gcc -c -x c -'
        elif lang == 'js':
            cmd = 'node -c -'
        elif lang == 'rb':
            cmd = 'ruby -c -'
        elif lang == 'jsx':
            cmd = 'NODE_PATH=$(npm root --quiet -g) npx babel --presets @babel/preset-react --no-babelrc'
        try:
            process = subprocess.run(cmd, shell=True, timeout=5, input=code.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if process.returncode == 0:
                return 0
            else:
                return 1
        except subprocess.TimeoutExpired:
            return 1
    elif lang in ('java', 'go'):
        with tempfile.NamedTemporaryFile(mode='w+', prefix='code', suffix='.'+lang, delete=False) as temp_file:
            temp_file_name = temp_file.name
            if lang == 'java':
                temp_file.write(code.replace('MyTestClass', os.path.basename(temp_file_name)[:-5]))
                cmd = 'javac {} {}'.format(get_cp_args(info), temp_file_name)
            elif lang == 'go':
                temp_file.write(code)
                cmd = f'gofmt {temp_file_name}'
        try:
            process = subprocess.run(cmd, shell=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if process.returncode == 0:
                return 0
            else:
                return 1
        except subprocess.TimeoutExpired:
            return 1
        finally:
            os.remove(temp_file_name)
    else:
        raise NotImplementedError()

def get_url_content(url):
    try:
        f = urlopen(Request(url, headers={
                'User-Agent':'Mozilla/5.0',
                'Content-Type':'application/json',
                'Accept':'application/json'
            })).read()
        return f.decode('utf-8')
    except HTTPError as e:
        if e.code == 429:
            time.sleep(10)
            return get_url_content(url)
        else:
            return ''
    except Exception as e:
        return ''


def compute_bleu_score(hyp: str, ref: str) -> float:
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.
    
    """Compute BLEU score between two strings using SacreBleu."""
    # Compute and return the BLEU score using SacreBleu
    warnings.filterwarnings("ignore")
    return sacrebleu.corpus_bleu(
        [hyp],
        [[ref]],
        smooth_method="exp",
        force=False,
        lowercase=False,
        use_effective_order=False,
    ).score


def inspect_cwe_dist(dataset):

    vul_types = np.unique([sample['vul_type'] for sample in dataset])
    langs = np.unique([sample['file_name'].split('.')[-1] for sample in dataset])

    stats = {lang: {vul: 0 for vul in vul_types} for lang in langs}

    for sample in dataset:
        lang = sample['file_name'].split('.')[-1]
        cwe = sample['vul_type']
        stats[lang][cwe] += 1
        

    data = [[vul] + [counts[vul] for counts in stats.values()] for vul in vul_types]
    print(tabulate(data, headers=['Vulnerability'] + list(langs), stralign='right', tablefmt='orgtbl'))    
