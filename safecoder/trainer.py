import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .utils import set_seed, load_model
from .timer import Timer
from .dataset import CodeDataset
from .constants import FUNC, GOOD, BAD

class LossDict:
    def __init__(self, keys):
        self.d = OrderedDict()
        self.keys = keys
        for key in keys:
            self.d[key] = list()

    def step(self, other):
        for k in other.d:
            self.d[k] += other.d[k]

    def pretty_print(self, args):
        p = []
        for k, l in self.d.items():
            if len(l) > 0:
                s = sum(l) / len(l) / args.grad_acc_steps
                p.append(f'{k}: {round(s, 6)}')
        return ', '.join(p)

    def clear(self):
        for key in self.keys:
            self.d[key].clear()

    def __getitem__(self, k):
        return self.d[k]

def token_weighted_loss(loss_type, inputs, targets, weights):
    if loss_type == 'ce':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'nll':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.NLLLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'ul':
        probs = F.softmax(inputs, dim=-1)
        probs = torch.gather(probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.clamp((1.0-probs), min=1e-5)
        loss = -torch.log(probs)
    elif loss_type == 'kl':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1))
        weights = weights.view(-1)
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        loss = loss_fct(inputs, targets)
        loss = loss.sum(dim=1)
    else:
        assert False

    loss = loss[weights != 0]
    return loss.mean()

def get_logits_from_lm(lm, inputs, control_ids):
    if control_ids is not None:
        past = lm.get_past_from_prefix(control_ids)
    else:
        past = None
    outputs = lm(inputs, past_key_values=past)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.dataset = None
        if self.args.sven:
            self.loss_keys = ['lm', 'contra', 'kl']
        else:
            self.loss_keys = ['func', 'pos', 'neg']
            if self.args.kl_loss_weight > 0:
                self.loss_keys.append('kl')

    def step(self, batch):
        loss_dict = LossDict(self.loss_keys)

        sample_types, inputs, weights = batch
        inputs = inputs.to(self.model.device)
        shift_inputs = inputs[..., 1:]
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:]
        outputs = self.model(inputs)
        shift_logits = outputs.logits[..., :-1, :]

        loss_total = 0.0
        for sample_type in sample_types:
            if sample_type == FUNC:
                loss = token_weighted_loss('ce', shift_logits, shift_inputs, shift_weights)
                loss_dict['func'].append(loss.item())
                loss_total += loss
            elif sample_type == GOOD:
                loss = self.args.loss_weight * token_weighted_loss('ce', shift_logits, shift_inputs, shift_weights)
                loss_dict['pos'].append(loss.item())
                loss_total += loss
            elif sample_type == BAD:
                loss = self.args.loss_weight * token_weighted_loss('ul', shift_logits, shift_inputs, shift_weights)
                loss_dict['neg'].append(loss.item())
                loss_total += loss
            else:
                assert False

            if (sample_type == GOOD or sample_type == BAD) and self.args.kl_loss_weight > 0:
                with torch.no_grad():
                    ref_outputs = self.ref_model(inputs)
                shift_ref_log_probs = F.log_softmax(ref_outputs.logits[..., :-1, :], dim=-1)
                shift_log_probs = F.log_softmax(shift_logits, dim=-1)
                loss = self.args.kl_loss_weight * token_weighted_loss('kl', shift_log_probs, shift_ref_log_probs, 1-shift_weights) / 1000
                loss_dict['kl'].append(loss.item())
                loss_total += loss

        return loss_total, loss_dict

    def sven_step(self, batch):
        loss_dict = LossDict(self.loss_keys)

        control_ids, inputs, weights = batch
        inputs = inputs.to(self.model.device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:].squeeze(0)
        control_ids = control_ids.to(self.model.device)
        control_ids -= 1

        correct_logits, correct_label_probs = get_logits_from_lm(self.model, inputs, control_ids)
        lm_loss = token_weighted_loss('ce', correct_logits, shift_inputs, shift_weights)
        loss_dict['lm'].append(lm_loss.item())

        incorrect_control_ids = -1 * (control_ids - 1)
        incorrect_logits, incorrect_label_probs = get_logits_from_lm(self.model, inputs, incorrect_control_ids)

        contrastive_probs = torch.stack((correct_label_probs, incorrect_label_probs), dim=1)
        contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
        contrastive_log_probs = torch.log(contrastive_probs)
        contrastive_labels = torch.zeros(shift_inputs.shape, dtype=torch.int64).to(self.model.device)
        contrastive_loss = token_weighted_loss('nll', contrastive_log_probs, contrastive_labels, shift_weights)
        contrastive_loss *= 4
        loss_dict['contra'].append(contrastive_loss.item())

        assert self.args.kl_loss_weight > 0
        correct_log_probs = F.log_softmax(correct_logits, dim=-1)
        self.model.eval()
        with torch.no_grad():
            ref_logits, _ = get_logits_from_lm(self.model, inputs, None)
        self.model.train()
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        kl_loss = token_weighted_loss('kl', correct_log_probs, ref_log_probs, 1-shift_weights)
        incorrect_log_probs = F.log_softmax(incorrect_logits, dim=-1)
        kl_loss += token_weighted_loss('kl', incorrect_log_probs, ref_log_probs, 1-shift_weights)
        kl_loss = kl_loss * self.args.kl_loss_weight / 1000
        loss_dict['kl'].append(kl_loss.item())

        loss_total = lm_loss + contrastive_loss + kl_loss

        return loss_total, loss_dict

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            loss, loss_dict = self.sven_step(batch) if self.args.sven else self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def load_model(self):
        self.tokenizer, self.model = load_model(self.args.pretrain_name, self.args)
        self.model.train()

        if self.args.kl_loss_weight > 0 and not self.args.sven:
            _, self.ref_model = load_model(self.args.pretrain_name, self.args)
            self.ref_model.eval()

    def load_dataset(self):
        self.dataset = CodeDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = CodeDataset(self.args, self.tokenizer, 'val')

    def save(self, path):
        """
        For normal models this saves the whole set of weights, for LoRA models it saves the adapter.
        """
        if self.args.sven:
            os.makedirs(path, exist_ok=True)
            prefix_file = os.path.join(path, 'pytorch_model.bin')
            state_dict = self.model.prefix_params.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            torch.save(state_dict, prefix_file)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def create_lora_config(self):
        """
        Includes all linear layers in the LoRA training.
        """
        self.lora_config = LoraConfig(
            r=self.args.r,
            target_modules=list(set([name for name in re.findall(r'\((\w+)\): Linear', str(self.model.modules))])),
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            task_type="CAUSAL_LM"
        )

    def run(self):
        self.load_model()
        self.load_dataset()

        if self.args.lora:
            self.create_lora_config()
            self.model = get_peft_model(self.model, self.lora_config)

        self.args.logger.info(f'Training args {self.args}')

        batch_size = self.args.batch_size
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        total_samples = len(self.dataset)
        batch_size = batch_size * self.args.grad_acc_steps
        total_steps = total_samples // batch_size * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps)
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.args.logger.info('***** Running training *****')
        self.args.logger.info('  Num samples = %d', total_samples)
        self.args.logger.info('  Num epoch = %d', self.args.num_train_epochs)
        self.args.logger.info('  Batch size= 1')
        self.args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
        self.args.logger.info('  Gradient Accumulation steps = %d', self.args.grad_acc_steps)
        self.args.logger.info('  Total optimization steps = %d', total_steps)
        self.args.logger.info('  Num val samples = %d', len(self.val_dataset))
        self.args.logger.info('  Num parameters = %d', num_params)
        self.args.logger.info('  Num trainable parameters = %d', num_trainable_params)

        global_step, acc_loss_dict = 0, LossDict(self.loss_keys)
        set_seed(self.args.seed)
        timer = Timer(total_steps)
        timer.start()
        self.model.train()
        for idx in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss, loss_dict = self.sven_step(batch) if self.args.sven else self.step(batch)
                loss /= self.args.grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                acc_loss_dict.step(loss_dict)

                if (step+1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()  
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        acc_loss_pp = acc_loss_dict.pretty_print(self.args)
                        self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s, %s', idx+1, self.args.num_train_epochs, global_step, total_steps, acc_loss_pp, timer)
                        acc_loss_dict.clear()

                    timer.end()
                    timer.start()

            if self.args.save_epochs > 0 and (idx+1) % self.args.save_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_loss_pp = self.do_eval()
                self.model.train()
                self.args.logger.info('val epoch %s: %s', idx+1, eval_loss_pp)
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.save(output_dir)
                self.save(last_output_dir)

        if (idx+1) % self.args.save_epochs != 0:
            self.model.eval()
            with torch.no_grad():
                eval_loss_pp = self.do_eval()
            self.args.logger.info('final eval loss: %s', eval_loss_pp)
            # output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
            last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
            # self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
            self.args.logger.info('Saving model checkpoint to %s', last_output_dir)
            # self.save(output_dir)
            self.save(last_output_dir)
