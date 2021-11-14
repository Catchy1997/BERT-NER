import pandas as pd
import os, json, re
import pickle as pkl
from tqdm import tqdm
import emoji


def print_dir(path_list, filepath, pattern):
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            print_dir(path_list, path, pattern)
        if path.endswith(pattern):
            path_list.append(path)

    return path_list


def query_normalization_new(query):    
    query = emoji.emojize(query, use_aliases=True)
    p = re.compile(r'@[:\u4E00-\u9FA5]+\([\w]*\)|@\w+\s*\(.*?\)\s*|@\s*', re.S)
    query = p.sub('', query)

    p = re.compile(r'\.|\n|\r|“|”|[\U00010000-\U0010ffff]|#|@|/|:|，|。|！|？|\"|!|\'', re.S)
    query = p.sub('', query)

    p = re.compile(r'\|', re.S)
    query = p.sub(' ', query)
    return query


def query_normalization(query):
    if query == '' or query is None:
        return ""

    re_tag = re.compile('@.*?\(.*?\)|#|@')
    query = re_tag.sub('', query)
    
    p = re.compile(r'[^\u4e00-\u9fa5^a-z^A-Z^0-9]|^[a-zA-Z]*$', re.S)
    query = p.sub('', query)
    
    return query.strip()


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


def normalization(query):
    if query == '' or query is None:
        return ""

    re_tag = re.compile('\t|\n|\r')
    query = re_tag.sub('', query)
    
    return query.strip()



if __name__ == "__main__":
    ### add ocr
    # with open('/home/zhanglin12/pyspark_script/data/ocr_dic_1kw.pkl','rb') as out_data:
    #     ocr_dic = pkl.load(out_data)
    # print('read pickle done: {}'.format(len(ocr_dic)))

    # for i in range(4, 5):
    #     with open('/home/zhanglin12/BERT-NER/datasets/query/part/part-'+str(i)+'.json', 'r') as f:
    #         for line in tqdm(f):
    #             data = json.loads(line)
    #             data['cover_ocr'] = ""

    #             if ocr_dic.get(str(data['photo_id'])) != None:
    #                 data['cover_ocr'] = ocr_dic[str(data['photo_id'])]
                
    #             with open('/home/zhanglin12/BERT-NER/datasets/query/part/part-'+str(i)+'-.json', 'a') as f:
    #                 f.write(json.dumps(data, ensure_ascii=False)+'\n')
    
    # for j in range(3, 7):
    #     with open('/home/zhanglin12/pyspark_script/data/ocr_dic_'+str(j)+'kw.pkl','rb') as out_data:
    #         ocr_dic = pkl.load(out_data)
    #     print('read pickle done: {}'.format(len(ocr_dic)))

    #     for i in range(3, 5):
    #         with open('/home/zhanglin12/BERT-NER/datasets/query/part/part-'+str(i)+'-'*(j-1)+'.json', 'r') as f:
    #             for line in tqdm(f):
    #                 data = json.loads(line)
    #                 if data['cover_ocr'] == "":
    #                     if ocr_dic.get(str(data['photo_id'])) != None:
    #                         data['cover_ocr'] = ocr_dic[str(data['photo_id'])]
                        
    #                 with open('/home/zhanglin12/BERT-NER/datasets/query/part/part-'+str(i)+'-'*j+'.json', 'a') as f:
    #                     f.write(json.dumps(data, ensure_ascii=False)+'\n')

    ### split
    # path_list = []
    # n = 0
    # path_list = print_dir(path_list, '/home/zhanglin12/BERT-NER/datasets/query/part', ".json")
    # for path in tqdm(path_list):
    #     with open(path, 'r') as f:
    #         for line in tqdm(f):
    #             n += 1

    #             line = json.loads(line)
    #             caption = caption_normalization(line['caption'])
    #             ocr = caption_normalization(line['ocr_result_new'])
    #             cover = caption_normalization(line['cover_ocr'])
    #             del line['ocr_result_new']
    #             line['caption_process'] = caption
    #             line['ocr_process'] = ocr
    #             line['cover_ocr_process'] = cover
                    
    #             with open('/home/zhanglin12/BERT-NER/datasets/query/test-'+str(n//1000000)+'.json', 'a') as f:
    #                 f.write(json.dumps(line, ensure_ascii=False)+'\n')


    """train"""
    ### csv to json
    # star_biaozhu = pd.read_csv('/home/zhanglin12/BERT-NER/datasets/query/data/star_biaozhu.csv')
    # star_keyword = pd.read_csv('/home/zhanglin12/BERT-NER/datasets/query/data/star_keyword.csv')
    # data = star_biaozhu.append(star_keyword)
    # data = data.drop_duplicates('photo_id')
    # print(len(data))
    # for i in tqdm(range(len(data))):
    #     line = json.loads(data.iloc[i].to_json())
    #     with open('/home/zhanglin12/BERT-NER/datasets/query/data/caption.json', 'a') as f:
    #         f.write(json.dumps(line, ensure_ascii=False)+'\n')

    # with open('/home/zhanglin12/BERT-NER/datasets/query/data/general_test.json', 'r') as f:
    #     json_data = f.readlines()
    # n = 0
    # for data in tqdm(json_data):
    #     data = json.loads(data)
    #     caption = caption_normalization(data['caption'])
    #     query = query_normalization(data['query'])

    #     if caption == '':
    #         continue
    #     if query == '':
    #         continue
    
    #     index = caption.find(query)
    #     if index == -1:
    #         continue
        
    #     ### 以下为query == caption[index:index+len(query)]的photo
    #     n += 1
    #     data['caption_pro'] = caption
    #     data['query_index'] = [index, index+len(query)]
    #     # data['query_index'] = ["O"] * len(caption)
    #     with open('/home/zhanglin12/BERT-NER/datasets/query/data/general_dev_query_index.json', 'a') as f:
    #         f.write(json.dumps(data, ensure_ascii=False)+'\n')
    # print('数据集数量={}'.format(n))
    # print('数据示例={}'.format(data))

    # ## get_label
    # id2label = {0: "O", 1: "B-QUERY", 2: "I-QUERY", 3: "S-QUERY", 4: "[START]", 5: "[END]", 6: "X"}
    # with open('/home/zhanglin12/BERT-NER/datasets/query/data/general_dev_query_index.json', 'r') as f:
    #     json_data = f.readlines()
    # for data in tqdm(json_data): # 10965
    #     data = json.loads(data)
    #     query_index = ["O"] * len(data['caption_pro'])
    #     for i, char in enumerate(data['caption_pro']):
    #         if i == data['query_index'][0]:
    #             query_index[i] = "B-QUERY"
    #         elif i > data['query_index'][0] and i <= data['query_index'][1]-1:
    #             query_index[i] = "I-QUERY"
    #         elif i == data['query_index'][1]-1:
    #             query_index[i] = "S-QUERY"
    #     data['query_index'] = query_index
    #     with open('/home/zhanglin12/BERT-NER/datasets/query/data/general_dev_query_index_final.json', 'a') as f:
    #         f.write(json.dumps(data, ensure_ascii=False)+'\n')


    # """test"""
    # with open('/home/zhanglin12/BERT-NER/datasets/query/data/general_test.json', 'r') as f:
    #     json_data = f.readlines()
    # n = 0
    # for data in tqdm(json_data):
    #     data = json.loads(data)
    #     caption = caption_normalization(data['caption'])

    #     if caption == '':
    #         continue
        
    #     n += 1
    #     data['caption_pro'] = caption
    #     data['query_index'] = ["O"] * len(caption)
    #     with open('/home/zhanglin12/BERT-NER/datasets/query/data/general_query_index.json', 'a') as f:
    #         f.write(json.dumps(data, ensure_ascii=False)+'\n')
    # print('数据集数量={}'.format(n))
    # print('数据示例={}'.format(data))

    ### compare
    with open('/home/zhanglin12/BERT-NER/datasets/query/v1.json', 'r') as f:
        v1_lines = f.readlines()
    print(len(v1_lines))
    with open('/home/zhanglin12/BERT-NER/datasets/query/v2.json', 'r') as f:
        v2_lines = f.readlines()
    print(len(v2_lines))
    with open('/home/zhanglin12/BERT-NER/datasets/query/test.json', 'r') as f:
        lines_ori = f.readlines()
    print(len(lines_ori))

    for i in tqdm(range(len(lines_ori))):
        v1_line, v2_line = json.loads(v1_lines[i]), json.loads(v2_lines[i])
        
        if v1_line['photo_id'] == v2_line['id']:
            query1 = query_normalization(v1_line['query'])
            if len(query1) <= 1:
                query1 = ''
            
            query2 = ''
            if v2_line['label'].get('QUERY') != None:
                for k in v2_line['label']['QUERY']:
                    query2 = k
            query2 = query_normalization(query2)

            result = 'False'
            if query2 == query1:
                result = 'True'
            
            with open('/home/zhanglin12/BERT-NER/datasets/query/result.csv', 'a') as f:
                f.write(str(v1_line['photo_id'])+'\t"'+normalization(v2_line['text'])+'"\t'+query1+'\t'+str(len(query1))+'\t'+query2+'\t'+str(len(query2))+'\t'+result+'\t'+json.loads(lines_ori[i])['category_name']+'\t'+str(json.loads(lines_ori[i])['category_id'])+'\n')
        