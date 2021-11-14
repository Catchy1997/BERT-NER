import os, re, time, copy, json
import pandas as pd
import logging, argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from models.transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
from models.transformers import BertModel, BertPreTrainedModel
from models.bert_for_ner import BertCrfForNer
from processors.utils_ner import get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from callback.progressbar import ProgressBar
from tools.common import seed_everything, json_to_text
from tools.common import init_logger, logger


def load_and_cache_examples(args, tokenizer, path, data_type='train'):
    # if args.local_rank not in [-1, 0] and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    processor = processors[args.task_name]()
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir+'/cache', 'cached_crf-{}'.format(args.input_dir.split('/')[-1].split('.')[0]))
    # cached_features_file = '/home/zhanglin12/BERT-NER/datasets/query_infer/generaljingxuan_caption_query_index'
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args)
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, label_list=label_list,
                                                max_seq_length=args.max_seq_length, cls_token_segment_id=0)
        if args.local_rank == -2:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(device)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long).to(device)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def predict(args, test_dataloader, model, tokenizer, photo_texts=None):    
    output_predict_file = os.path.join(args.output_dir, args.input_dir.split('/')[-1])
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    results = []

    if args.local_rank != -1 and args.n_gpu > 1:
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad(): 
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask']).to(device)
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        
        ## one batch
        for i in range(len(tags)):
            preds = tags[i][1:-1]  # [CLS]XXXX[SEP]
            label_entities = get_entities(preds, args.id2label, args.markup)
            json_d = {}
            json_d['id'] = step
            json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
            json_d['entities'] = label_entities
            results.append(json_d)
        pbar(step)
        with open(output_predict_file, 'a') as f:
            for record in results:
                f.write(json.dumps(record) + '\n')
        results = []
    logger.info("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--best_model', type=str)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--max_seq_length', default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument('--overwrite_output_dir', type=bool)
    parser.add_argument('--id2label', type=dict)
    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'])
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--task_name", default='query', type=str, required=False, help="The name of the task to train selected in the list: ")
    parser.add_argument('--str_name', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    args.id2label = {0: "O", 1: "B-QUERY", 2: "I-QUERY", 3: "S-QUERY", 4: "[START]", 5: "[END]", 6: "X"}

    
    ### set random_seed
    seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    
    ### Setup CUDA, GPU & distributed training
    if args.local_rank in [1, 0]:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        # torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    
    ### dataloader
    logger.info("Training/evaluation parameters %s", args)

    config = BertConfig.from_pretrained(args.base_path, num_labels=args.num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.base_path, do_lower_case=True)

    test_dataset = load_and_cache_examples(args, tokenizer, path=args.input_dir, data_type='test')
    
    logger.info("  Num examples = %d", len(test_dataset))
    # Note that DistributedSampler samples randomly
    # test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_sampler = SequentialSampler(test_dataset) if -1 == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=args.batch_size, collate_fn=collate_fn)


    ## load_model
    model = BertCrfForNer.from_pretrained(args.base_path, config=config)

    pretrain_model = torch.load(args.best_model)
    del pretrain_model['bert.embeddings.position_ids']
    model.load_state_dict(pretrain_model)
    model = model.to(device)

    if args.local_rank != -1 and args.n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
    
    ### predict
    logger.info("***** Running prediction %s *****", 'bert_model')
    logger.info("  Batch size = %d", args.batch_size)
    predict(args, test_dataloader, model, tokenizer, photo_texts=None)