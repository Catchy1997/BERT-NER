import json, os
from tqdm import tqdm
from tools.common import seed_everything, json_to_text
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--text_path', type=str)
    parser.add_argument('--pred_str_name', type=str)
    parser.add_argument('--true_str_name', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    results = []
    with open(args.pred_path, 'r') as fr:
        for line in fr:
            results.append(json.loads(line))
    print('READ pred DONE')
    
    test_text = []
    with open(args.text_path, 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    print('READ text DONE')
    
    test_submit = []
    for x, y in tqdm(zip(test_text, results)):
        json_d = {}
        json_d['id'] = x['photo_id']
        json_d['text'] = x[args.true_str_name]
        json_d['query'] = x['query']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x[args.pred_str_name])
        if len(entities) != 0:
            for subject in entities[:1]:
                tag = subject[0]
                start = subject[1]
                # end = subject[2] + 1
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    
    print('Start write.....')
    json_to_text(args.save_path, test_submit)

    n = 0
    with open(args.save_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            print(data)
            if data['label'].get('QUERY') != None:
                print(data['id'], [x for x in data['label']['QUERY'].keys()])
            n += 1
            if n > 5:
                break
    print(n)
