import torch
import os, json, re
from tqdm import tqdm
import numpy as np 
from inferSDK import InferKessContext

from models.transformers import BertTokenizer, BertConfig
from processors.utils_ner import get_entities
from models.bert_for_ner import BertCrfForNer


device = torch.device("cuda")
    
def rpc_model_predict(inputs, photo_text):
    model_name = "query-bert-ner"
    model_version = -1
    service_name = "grpc_search-tech_query-bert-ner-0712"
    kess_division = "central"
    ctx = InferKessContext(model_name, model_version, service_name,kess_division)
    model_inputs = convert_text_to_tensor(text=photo_text, tokenizer=tokenizer)
    outputs = {}
    outputs["OUTPUT__0"] = np.float32
    
    results = ctx.RunSync(model_inputs, outputs)

    tags = []
    for name, value in results.items():
        value = value.reshape(1, 128, 8)
        preds = np.argmax(value, axis=2).tolist()
        preds = preds[0][1:-1]
        tags = [ID2LABEL[x] for x in preds]
    label_entities = get_entities(preds, ID2LABEL, MARTKUP)
    text = ""
    if len(label_entities)>0:
        st_idx = label_entities[0][1]
        ed_idx = label_entities[0][2]
        text = photo_text[st_idx: ed_idx+1]
        print('text: ', text)
    return text

    
def convert_text_to_tensor(text, tokenizer, max_seq_length=128, cls_token="[CLS]", cls_token_segment_id=0,
                            sep_token="[SEP]", pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
    tokens = tokenizer.tokenize(text)
    input_len = len(text)
    # Account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
    #  Input_query:
    #  这种随机超能力你敢尝试吗？动作科幻电影《超能计划》
    #  preprocess: 
    #  tokens:   [CLS] 这 种 随 机 超 能 力 你 敢 尝 试 吗 ？动 作 科 幻 电 影 《 超 能 计 划 》[SEP]
    #  input_ids: 101 6821 4905 7390 3322 6631 5543 1213 872 3140 2214 6407 1408 8043 1220 868 4906 2404 4510 2512 517 6631 5543 6369 1153 518 102 0 0
    #  segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0

    tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. 
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    # Zero-pad up to the sequence length.
        
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token] * padding_length)
    input_mask += ([0 if mask_padding_with_zero else 1] * padding_length) 
    segment_ids += ([pad_token_segment_id] * padding_length)
        
    model_inputs = {}
    input_ids = np.asarray(input_ids, dtype=np.int64)
    model_inputs["INPUT__0"] = input_ids[np.newaxis, :]
    # print('input_ids: ', model_inputs["input_ids"].shape)
        
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    model_inputs["INPUT__1"] = segment_ids[np.newaxis, :]
    # print('input_ids: ', model_inputs["token_type_ids"].shape)
        
    input_mask = np.asarray(input_mask, dtype=np.int64)
    model_inputs["INPUT__2"] = input_mask[np.newaxis, :]
    # print('attention_mask: ', model_inputs["attention_mask"].shape)
        
    return model_inputs


def preprocess_text_to_tensor(text, tokenizer, max_seq_length=128, cls_token="[CLS]", cls_token_segment_id=0,
                            sep_token="[SEP]", pad_token=0, pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
    tokens = tokenizer.tokenize(text)
    # Account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
    #  Input_query:
    #  这种随机超能力你敢尝试吗？动作科幻电影《超能计划》
    #  preprocess: 
    #  tokens:   [CLS] 这 种 随 机 超 能 力 你 敢 尝 试 吗 ？动 作 科 幻 电 影 《 超 能 计 划 》[SEP]
    #  input_ids: 101 6821 4905 7390 3322 6631 5543 1213 872 3140 2214 6407 1408 8043 1220 868 4906 2404 4510 2512 517 6631 5543 6369 1153 518 102 0 0
    #  segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0

    tokens += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. 
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token] * padding_length)
    input_mask += ([0 if mask_padding_with_zero else 1] * padding_length) 
    segment_ids += ([pad_token_segment_id] * padding_length)
        
    # model_inputs = {}
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    # model_inputs['input_ids'] = input_ids
    token_type_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(device)
    # model_inputs['token_type_ids'] = token_type_ids
    attention_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0).to(device)
        
    # model_inputs['attention_mask'] = attention_mask
    return input_ids, token_type_ids, attention_mask


def caption_normalization(query):    
    if query == '' or query is None:
        return ""

    re_tag = re.compile('@.*?\(.*?\)|#|@')
    query = re_tag.sub('', query)
    
    # p = re.compile(r'\n|\r|\t|!|！| |  |\?|？|\"|\'|\|“|”|《|》|：|,|\.|\xa0|-|“', re.S)
    p = re.compile(r',|，|。|！|：', re.S)
    query = p.sub('zzy', query)
    p = re.compile(r'[^\u4e00-\u9fa5^a-z^A-Z^0-9]|^[a-zA-Z]*$', re.S)
    query = p.sub('', query)
    p = re.compile(r'zzy', re.S)
    query = p.sub('，', query)
    
    return query.strip()


def query_normalization(query):
    if query == '' or query is None:
        return ""

    re_tag = re.compile('@.*?\(.*?\)|#|@')
    query = re_tag.sub('', query)
    
    p = re.compile(r'[^\u4e00-\u9fa5^a-z^A-Z^0-9]|^[a-zA-Z]*$', re.S)
    query = p.sub('', query)
    
    return query.strip()


def normalization(query):
    if query == '' or query is None:
        return ""

    re_tag = re.compile('\t|\n|\r')
    query = re_tag.sub('', query)
    
    return query.strip()


if __name__ == "__main__":
    BASE_PATH = 'prev_trained_model/bert-base-chinese'
    NUM_LABELS = 7
    ID2LABEL = {0: 'O', 1: 'B-QUERY', 2: 'I-QUERY', 3: 'S-QUERY', 4: '[START]', 5: '[END]', 6: 'X'}
    MARTKUP = 'bios'
    config = BertConfig.from_pretrained(BASE_PATH, num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained(BASE_PATH, do_lower_case=True)
    # traced_model = torch.jit.load('best_model_test.pt')
    
    # text = "陈卓璇说丁太升的吐槽音色和flow一般最后当面灭灯也太可爱了吐槽大会我在快手追综艺"
    text = '杭州近年来最年轻的校长来了95后入职才4年当上中学副校长'
    model_inputs = convert_text_to_tensor(text, tokenizer)
    res = rpc_model_predict(model_inputs, text)
    print(res)