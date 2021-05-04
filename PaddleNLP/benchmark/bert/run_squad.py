# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import os
import random
import time

from functools import partial
import numpy as np
import paddle
import distutils.util

import argparse
from paddle.io import DataLoader

import paddlenlp as ppnlp

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer, ErnieForQuestionAnswering, ErnieTokenizer
from paddlenlp.metrics.squad import squad_evaluate, compute_predictions

from paddle.fluid.contrib.sparsity import ASPHelper, check_mask_2d, check_mask_1d
from paddle.fluid import global_scope
from paddle.fluid.io import load_vars

MODEL_CLASSES = {
    "bert": (BertForQuestionAnswering, BertTokenizer),
    "ernie": (ErnieForQuestionAnswering, ErnieTokenizer)
}

class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=start_logits, label=start_position, soft_label=False)
        start_loss = paddle.mean(start_loss)
        end_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=end_logits, label=end_position, soft_label=False)
        end_loss = paddle.mean(end_loss)

        loss = (start_loss + end_loss) / 2
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Directory of all the data for train, valid, test.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Type of pre-trained model.")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name of model.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.0,
        type=float,
        help="Proportion of training steps to perform linear learning rate warmup for."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument(
        "--max_query_length", type=int, default=64, help="Max query length.")
    parser.add_argument(
        "--max_answer_length", type=int, default=30, help="Max answer length.")
    parser.add_argument(
        "--do_lower_case",
        action='store_false',
        help="Whether to lower case the input text. Should be True for uncased models and False for cased models."
    )
    parser.add_argument(
        "--verbose", action='store_true', help="Whether to output verbose log.")
    parser.add_argument(
        "--version_2_with_negative",
        action='store_true',
        help="If true, the SQuAD examples contain some that do not have an answer. If using squad v2.0, it should be set true."
    )
    parser.add_argument(
        "--select_device",
        type=str,
        default="gpu",
        help="Device for selecting for the training.")
    parser.add_argument(
        "--sparsity",
        default=False,
        type=bool,
        help="True for enabling ASP.",
    )
    parser.add_argument(
        "--load_dir",
        default=None,
        type=str,
        required=False,
        help="The directory where the model predictions and checkpoints will be loaded.",
    )
    parser.add_argument(
        "--nonprune",
        default=False,
        type=bool,
        help="True for pruning models in ASP.",
    )
    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=1.0,
        help="The value of scale_loss for fp16.")
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

def create_data_holder():
    input_ids = paddle.static.data(
        name="input_ids", shape=[-1, -1], dtype="int64")
    segment_ids = paddle.static.data(
        name="segment_ids", shape=[-1, -1], dtype="int64")
    start_positions = paddle.static.data(
        name="start_positions", shape=[-1, 1], dtype="int64")
    end_positions = paddle.static.data(
        name="end_positions", shape=[-1, 1], dtype="int64")
    unique_id = paddle.static.data(
        name="unique_id", shape=[-1, 1], dtype="int64")
    return input_ids, segment_ids, start_positions, end_positions, unique_id

def reset_program_state_dict(args, model, state_dict, pretrained_state_dict):
    reset_state_dict = {}
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else getattr(model, args.model_type).config["initializer_range"]
    for n, p in state_dict.items():
        if n not in pretrained_state_dict:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            reset_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
        else:
            reset_state_dict[p.name] = pretrained_state_dict[n]
    return reset_state_dict

def evaluate(exe, logits, dev_program, data_loader, args):
    RawResult = collections.namedtuple(
        "RawResult", ["unique_id", "start_logits", "end_logits"])

    all_results = []
    tic_eval = time.time()

    for batch in data_loader:
        start_logits_tensor, end_logits_tensor = exe.run(dev_program, feed=batch, fetch_list=[*logits])

        unipue_ids = np.array(batch[0]['unique_id'])
        for idx in range(unipue_ids.shape[0]):
            if len(all_results) % 1000 == 0 and len(all_results):
                print("Processing example: %d" % len(all_results))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()
            unique_id = int(unipue_ids[idx])
            start_logits = [float(x) for x in start_logits_tensor[idx]]
            end_logits = [float(x) for x in end_logits_tensor[idx]]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

    all_predictions, all_nbest_json, scores_diff_json = compute_predictions(
        data_loader.dataset.examples, data_loader.dataset.features, all_results,
        args.n_best_size, args.max_answer_length, args.do_lower_case,
        args.version_2_with_negative, args.null_score_diff_threshold,
        args.verbose, data_loader.dataset.tokenizer)

    squad_evaluate(data_loader.dataset.examples, all_predictions,
                   scores_diff_json, 1.0)

def do_train(args):
    # Set the paddle execute enviroment
    paddle.enable_static()
    place = paddle.set_device(args.select_device)
    set_seed(args)

    # Create the main_program for the training and dev_program for the validation
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    root = args.data_path

    train_dataset = ppnlp.datasets.SQuAD(
        tokenizer=tokenizer,
        doc_stride=args.doc_stride,
        root=root,
        version_2_with_negative=args.version_2_with_negative,
        max_query_length=args.max_query_length,
        max_seq_length=args.max_seq_length,
        mode="train")

    train_batch_sampler = paddle.io.BatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    train_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack(),  # unipue_id
        Stack(dtype="int64"),  # start_pos
        Stack(dtype="int64")  # end_pos
    ): [data for i, data in enumerate(fn(samples)) if i != 2]

    with paddle.static.program_guard(main_program, startup_program):
        input_ids, segment_ids, start_positions, end_positions, unique_id = create_data_holder()

    train_data_loader = DataLoader(
        dataset=train_dataset,
        feed_list=[input_ids, segment_ids, start_positions, end_positions],
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        num_workers=0,
        return_list=False)

    dev_dataset = ppnlp.datasets.SQuAD(
        tokenizer=tokenizer,
        doc_stride=args.doc_stride,
        root=root,
        version_2_with_negative=args.version_2_with_negative,
        max_query_length=args.max_query_length,
        max_seq_length=args.max_seq_length,
        mode="dev")

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_dataset, batch_size=args.batch_size, shuffle=False)

    dev_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack()  # unipue_id
    ): fn(samples)

    dev_data_loader = DataLoader(
        dataset=dev_dataset,
        feed_list=[input_ids, segment_ids, unique_id],
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        num_workers=0,
        return_list=False)

    with paddle.static.program_guard(main_program, startup_program):
        model, pretrained_state_dict = model_class.from_pretrained(args.model_name_or_path)
        criterion = CrossEntropyLossForSQuAD()
        logits = model(input_ids=input_ids, token_type_ids=segment_ids)
        dev_program = main_program.clone(for_test=True)

    with paddle.static.program_guard(main_program, startup_program):
        loss = criterion(logits, (start_positions, end_positions))
        lr_scheduler = paddle.optimizer.lr.LambdaDecay(
            args.learning_rate,
            lambda current_step, warmup_proportion=args.warmup_proportion,
            num_training_steps=args.max_steps if args.max_steps > 0 else
            (len(train_dataset.examples)//args.batch_size*args.num_train_epochs): float(
                current_step) / float(max(1, warmup_proportion*num_training_steps))
            if current_step < warmup_proportion*num_training_steps else max(
                0.0,
                float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - warmup_proportion*num_training_steps))))

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in [
                p.name for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ])

        if args.use_amp:
            amp_list = paddle.fluid.contrib.mixed_precision.AutoMixedPrecisionLists(
                custom_white_list=['softmax', 'layer_norm', 'gelu'])
            optimizer = paddle.fluid.contrib.mixed_precision.decorate(
                optimizer,
                amp_list,
                init_loss_scaling=args.scale_loss,
                use_dynamic_loss_scaling=True)
        if args.sparsity:
            ASPHelper.set_excluded_layers(main_program, ['linear_72', 'linear_73'])
            optimizer = ASPHelper.decorate(optimizer)
        optimizer.minimize(loss)

    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    state_dict = model.state_dict()
    reset_state_dict = reset_program_state_dict(args, model, state_dict,
                                                pretrained_state_dict)
    paddle.static.set_program_state(main_program, reset_state_dict)

    if args.load_dir is not None:
        print("-------------------- Loading model --------------------")
        print("Load model weights from:", args.load_dir)
        vars=ASPHelper.get_vars(main_program)
        if args.nonprune:
            vars = main_program.global_block().all_parameters()
        load_vars(exe, args.load_dir, main_program, vars=vars)
        if args.sparsity and args.nonprune:
            for param in main_program.global_block().all_parameters():
                if ASPHelper.is_supported_layer(main_program, param.name):
                    mat = np.array(global_scope().find_var(param.name).get_tensor())
                    assert check_mask_1d(mat.T, 4, 2), "{} is not in 2:4 sparse pattern".format(param.name)
        print("-------------------- Loading model Done ---------------")

    if args.sparsity and (not args.nonprune):
        print("-------------------- Sparsity Pruning --------------------")
        ASPHelper.prune_model(place, main_program)
        print("-------------------- Sparsity Pruning Done ---------------")
        # evaluate(exe, logits, dev_program, dev_data_loader, args)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            loss_return = exe.run(main_program, feed=batch, fetch_list=[loss])

            lr_scheduler.step()

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss_return[0],
                        args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

        output_dir = os.path.join(args.output_dir,
                                    "model_%d" % epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        paddle.fluid.io.save_params(exe, output_dir)
        tokenizer.save_pretrained(output_dir)

        evaluate(exe, logits, dev_program, dev_data_loader, args)

    output_dir = os.path.join(args.output_dir, "model_final")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    paddle.fluid.io.save_params(exe, output_dir)
    tokenizer.save_pretrained(output_dir)

    if args.sparsity:
        print("-------------------- Sparsity Checking --------------------")
        for param in main_program.global_block().all_parameters():
            # if ASPHelper.is_supported_layer(param.name):
            mat = np.array(global_scope().find_var(param.name).get_tensor())
            print(param.name, "is 2:4 sparse pattern?", check_mask_1d(mat.T, 4, 2))
            # if not check_mask_1d(mat.T, 4, 2):
            #         print("!!!!!!!!!!", param.name, "Not in 2:4 Sparsity Validation")

if __name__ == "__main__":
    args = parse_args()
    do_train(args)
