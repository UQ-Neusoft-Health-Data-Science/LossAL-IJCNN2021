import re
import re
import json
import gzip
import os
import glob
# from pyumls import api
import utils.file_util as file_util
import utils.text_util as text_util
import time
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from scipy import sparse
# from pattern3.text.en import singularize
from nltk.stem import WordNetLemmatizer
import utils.text_util
import math
stop_words=list(set(stopwords.words('english')))
stop_words=[x for x in stop_words]
stop_words.append('a')
stop_words.append('an')
stop_words.append('the')
stop_words.append('&quot;')#"
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
# print(stop_words)
wnl = WordNetLemmatizer()
api_key='1bd1f2c2-fbab-4d54-aed9-fb442791d49d'
import pandas as pd
import numpy as np
def convert2csc1(info_data):
    src_ids=[]
    dst_ids=[]
    values=[]
    for row_num, row_data in enumerate(info_data):
        for column_num, column_data in enumerate(row_data):
            if column_data != 0:
                src_ids.append(row_num)
                dst_ids.append(column_num)
                values.append(column_data)
        print(row_num,'/',len(info_data))
    info_matrix = csc_matrix((values, (src_ids, dst_ids)), shape=info_data.shape)  # .toarray()# shape=(len(sources),len(dests))
    return info_matrix.nonzero()


def update_csc_matrix(count,src_ids,dst_ids,values,new_src,new_dst,new_val):
    src_ids.extend(new_src)
    dst_ids.extend(new_dst)
    values.extend(new_val)
    print(count)
def convert2csc2(info_data,pattern="sim"):

    src_ids = []
    dst_ids = []
    values = []
    # [src_ids.extend(len([y for y in enumerate(x.flat) if y != 0]) * [i])  for i, x in enumerate(np.array(info_data))]
    # [dst_ids.extend([j for j, y in enumerate(x.flat) if y != 0]) for i, x in enumerate(np.array(info_data))]
    # [values.extend([y for y in x.flat if y != 0]) for i, x in enumerate(np.array(info_data))]
    # file_util.dump(src_ids,"src_ids.pck")
    # file_util.dump(dst_ids, "dst_ids.pck")
    # file_util.dump(values, "values.pck")
    # print(np.array(src_ids).shape(),np.array(dst_ids.shape()), np.array(values.shape()))
    # info_matrix = csc_matrix((values, (src_ids, dst_ids)),
    #                          shape=(len(src_ids),len(dst_ids)))  # .toarray()# shape=(len(sources),len(dests))
    # return info_matrix.nonzero()
    [update_csc_matrix(i,src_ids,dst_ids,values,len([y for y in x.flat ]) * [i],[j for j,y in enumerate(x.flat) ],[y for y in x.flat ]) for i, x in enumerate(np.array(info_data))]
    # return sparse.csr_matrix(info_data)
    spmat= coo_matrix((values, (src_ids, dst_ids)), shape=(len(info_data),len(info_data[0])))
    return spmat,src_ids,dst_ids,values
def convert2csc3(info_data,pattern="sim"):
    return sparse.csr_matrix(info_data)
def convert_name_mention_UMLS_concepts(name_mention_list,start_index):
    name_mention_dict={}
    not_found_mentions=[]
    start_date=time.time()
    i=start_index
    while(i<len(name_mention_list)):
        name_mention=name_mention_list[i]
        print(i, '/', len(name_mention_list), ':', name_mention)
    # for i,name_mention in enumerate(name_mention_list):
        end_date=time.time()
        day_ticket, auth_client = api.getUTS1dayTicket(api_key)
        if (end_date-start_date)>900:
            day_ticket, auth_client = api.getUTS1dayTicket(api_key)
        umls_results = api.search_v1(name_mention, auth_client,day_ticket, version='2019AB', max_pages=5)
        for result in umls_results:
            current_cui = result['ui']
            if result.get('ui',None):
                if name_mention_dict.get(current_cui,None):
                    name_mention_dict[current_cui]['name_mentions'].append(name_mention)
                else:
                    name_mention_dict[current_cui]={'name':result['name'],'name_mentions':[name_mention]}
            else:
                not_found_mentions.append(name_mention)
                file_util.dump(not_found_mentions,"/home/s4616573/pubmed/pubmed_distill/not_found_mention"+str(start_index)+".pck")
                file_util.dump(not_found_mentions, "/home/s4616573/pubmed/pubmed_distill/not_found_mention"+str(start_index)+".json")
        file_util.dump(name_mention_dict,"/home/s4616573/pubmed/pubmed_distill/name_mention_cui"+str(start_index)+".pck")
        file_util.dump_json(name_mention_dict,"/home/s4616573/pubmed/pubmed_distill/name_mention_cui"+str(start_index)+".json")
        i+=1
def get_all_entities_in_ner_format(input_file):
    """Read a BIO data!"""
    # stop_words=open("/home/s4616573/code/bert/storage/stop_words.txt",'r').readlines()
    # stop_words=[re.sub('\n','',x) for x in stop_words]
    stop_words=['so', 'A',  'those', '&quot;', 'The', 're', 'your', 'after', 'such', 'these', 'not', 'An', 'this', 'Some', 'above', 'all', 'Her', 'any', 'an', 'more', 'THE', 'very', 'ALL', 'Our', 'AS', 'his', 'OTHER', 'All', 'too', 'Very', 'This', 'both',  'the', 'from', 'most', 'only', 'Each', 'Few', 'being', 'their', 'other',  'out',  'DO',  'That', 'His', 'that', 'her', 'Other', 'These', 'Most', 'Further',  'our', 'a', 'He', 'HER', 'few', 'some']

    # print(stop_words)
    removed_words=[]
    rf = open(input_file, 'r')
    lines = [];
    words = [];
    labels = []

    entity_words=[]
    entity_list=[]
    for line in rf:
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        # if word=='surgeries':
        #     print("debug")
        # here we dont do "DOCSTART" check
        if (len(line.strip()) == 0 and words[-1] == '.') or ( label=='O' ):
            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            lines.append((l, w))
            if len( entity_words):
                if entity_words[-1].endswith('s'):
                    entity_words[-1]=wnl.lemmatize(entity_words[-1])
                if entity_words[0].lower() in stop_words or entity_words[0].isnumeric():
                    current_entity=" ".join(entity_words[1:])
                    removed_words.append(entity_words[0])
                elif entity_words[-1].lower() in stop_words:
                    current_entity = " ".join(entity_words[0:len(entity_words)-1])
                    removed_words.append(entity_words[-1])
                else:
                    current_entity = ' '.join([word for word in entity_words if len(word) > 0])
                # current_entity=re.sub('\s{2,}',' ',current_entity)
                current_entity=current_entity.strip()
                if current_entity!='':
                    current_entity=re.sub('_','',current_entity)
                    current_entity=re.sub('&apos;',"'",current_entity)
                    entity_list.append(current_entity)
            words = []
            labels = []
            entity_words = []

        words.append(word)
        labels.append(label)
        if (label.startswith('B') or label.startswith('I')):
            entity_words.append(word)
    # print(list(set(removed_words)))
    rf.close()
    return list(set(entity_list))
    # return entity_list


def convert_data_ner_format(file_path,data_path="/home/s4616573/data/i2b2/",pattern_file="i2b2",is_i2b2=True):
    lines=open(file_path).readlines()
    # data_path="/home/s4616573/code/tf_ner/data/example/"
    # data_path="/home/s4616573/data/i2b2/"

    """Read a BIO data!"""
    rf = open(file_path, 'r')
    lines = [];
    words = [];
    labels = []
    text_lines = []
    label_lines = []

    for line in rf:
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        # here we dont do "DOCSTART" check
        if is_i2b2:
            if len(line.strip()) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])+'\n'
                lines.append((l, w))
                text_lines.append(w)
                words = []
                labels = []
        else:
            if len(line.strip()) == 0:
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])+'\n'
                lines.append((l, w))
                text_lines.append(w)
                words = []
                labels = []
        words.append(word)
        labels.append(label)
    rf.close()
    # return text_lines, label_lines
    if "train" in file_path:
        new_tag_file=open(data_path+pattern_file+"_train.tags.txt","w+")
        new_word_file = open(data_path +pattern_file+ "_train.words.txt", "w+")
    else:
        new_tag_file = open(data_path +pattern_file+ "_test.tags.txt", "w+")
        new_word_file = open(data_path + pattern_file+"_test.words.txt", "w+")
    new_word_file.writelines(text_lines)
    new_tag_file.writelines(label_lines)

    new_word_file.close()
    new_tag_file.close()
    # lines=open("/home/s4616573/code/tf_ner/data/example/i2b2_train.words.txt","r").readlines()
    lines = open(data_path+pattern_file+"_train.words.txt", "r").readlines()
    print("line count",len(lines))



def merge_CUI_pck(CUI_dir,file_pattern):
    CUI_list=[]
    CUI_distill_dir="/home/s4616573/pubmed/pubmed_distill/"
    if not os.path.exists(CUI_distill_dir):
        os.mkdir(CUI_distill_dir)
    for file_name in glob.iglob(CUI_dir + '**/*_'+file_pattern):
        print (file_name)
        CUI_arr=file_util.load(file_name)
        CUI_list.extend(list(CUI_arr))
        print("LEN CUIs",len(CUI_list))
    CUI_list=list(set(CUI_list))
    CUI_list=[re.sub('\'','',x) for x in CUI_list]
    file_util.dump(CUI_list,CUI_distill_dir+"all_CUIs.pck")
    return CUI_list

def update_CUI_all_infos(CUI_file,api_key,begin_idx,file_pattern):
    CUI_distill_dir = "/home/s4616573/pubmed/pubmed_distill/"
    CUI_list=file_util.load(CUI_file)
    CUI_dict={}
    semantic_type_dict={}
    concept_dict={}
    len_CUI_list=len(CUI_list)
    start_date=time.time()
    day_ticket,auth_client=api.getUTS1dayTicket(api_key)
    i=begin_idx
    while (i< len(CUI_list)):
        print("end_index",i)
        CUI=CUI_list[i]
        end_date=time.time()
        if (end_date-start_date)>3600*2:
            day_ticket, auth_client = api.getUTS1dayTicket(api_key)
        print("CUI:",i,"/",len_CUI_list,":",CUI)
        cui_info = api.getByCUI(CUI, day_ticket,auth_client)
        process_start_time=time.time()
        if cui_info:
            CUI_dict[CUI]={"name":cui_info["name"],"sem_type_uis":[],"definitions":[]}
            sem_types=cui_info.get("semanticTypes",[])
            concept_dict[cui_info["name"]]=CUI
            cui_def_link=cui_info.get("definitions","NONE")
            if cui_def_link!="NONE":
                cui_definitions=api.getDefinitionByCUI(cui_def_link, day_ticket,auth_client)
                for cui_def in cui_definitions:
                    CUI_dict[CUI]["definitions"].append(cui_def["value"])
            # for x in sem_types:
            #     sem_type_info=api.getSemanticTypeByCUI(x['uri'],day_ticket,auth_client)
            #     sem_type_name=sem_type_info["name"]
            #     sem_type_def=sem_type_info['definition']
            #     sem_type_ui=sem_type_info['ui']#T047 ui T047
            #     sem_type_abbr=sem_type_info['abbreviation']
            #     sem_type_group=sem_type_info['semanticTypeGroup']['expandedForm']
            #     semantic_type_dict[sem_type_ui]={"definition":sem_type_def,"abbreviation":sem_type_abbr,"group":sem_type_group}
            #     CUI_dict[CUI]["sem_type_uis"].append(sem_type_ui)
            #     concept_dict[sem_type_name]=sem_type_ui
            #     concept_dict[sem_type_abbr] = sem_type_ui
            # file_util.dump(concept_dict, CUI_distill_dir + "all_concept_dict"+file_pattern+".pck")
            # file_util.dump(semantic_type_dict, CUI_distill_dir + "semantic_type_dict"+file_pattern+".pck")
            # file_util.dump(CUI_dict, CUI_distill_dir + "CUI_dict"+file_pattern+".pck")
            file_util.dump_json(concept_dict, CUI_distill_dir + "all_concept_dict"+file_pattern+".json")
            # file_util.dump_json(semantic_type_dict, CUI_distill_dir + "semantic_type_dict"+file_pattern+".json")
            file_util.dump_json(CUI_dict, CUI_distill_dir + "CUI_dict"+file_pattern+".json")
            process_end_time = time.time()
            print("Process time",process_end_time-process_start_time)
        else:
            print("ERROR LINK","end_index")
        i+=1
    # file_util.dump(concept_dict,CUI_distill_dir+"all_concept_dict.pck")
    # file_util.dump(semantic_type_dict, CUI_distill_dir+"semantic_type_dict.pck")
    # file_util.dump(CUI_dict, CUI_distill_dir + "CUI_dict.pck")
    # file_util.dump_json(concept_dict, CUI_distill_dir + "all_concept_dict.json")
    # file_util.dump_json(semantic_type_dict, CUI_distill_dir + "semantic_type_dict.json")
    # file_util.dump_json(CUI_dict, CUI_distill_dir + "CUI_dict.json")

# merge_CUI_pck("/home/s4616573/pubmed/data_extract",'CUI')
# all_entities=get_all_entities_in_ner_format("/home/s4616573/data/i2b2/train.txt")
# ngram_mentions=text_util.generate_n_gram_from_name_mentions(all_entities,4)
# convert_name_mention_UMLS_concepts(ngram_mentions,9189)
# for entity in all_entities:
#     # if '_' in entity:
#         print(entity)
# print("entities",len(all_entities))
# file_util.dump(all_entities,"/home/s4616573/pubmed/pubmed_distill/i2b2_entities.pck")
# all_entities=[x+'\n' for x in all_entities]
# with open("/home/s4616573/pubmed/pubmed_distill/i2b2_entities.txt",'w') as f:
#     f.writelines(all_entities)
# n_gram_name_mentions = text_util.generate_n_gram_from_name_mentions(all_entities, 4)
# print("ngram",len(n_gram_name_mentions))
# file_util.dump(n_gram_name_mentions,"/home/s4616573/pubmed/pubmed_distill/i2b2_ngram_entities.pck")
# result=api.search("Tumor Mass", '1bd1f2c2-fbab-4d54-aed9-fb442791d49d', version='2019AB', max_pages=5)
# print(result)
# print(wnl.lemmatize("patches"))
# update_CUI_all_infos("/home/s4616573/pubmed/pubmed_distill/all_CUIs.pck",'1bd1f2c2-fbab-4d54-aed9-fb442791d49d',0,"_onlyCUI")
# print(re.sub('\'','',"'C0441655'"))
# convert_data_ner_format("/home/s4616573/data/i2b2/train.txt")
# convert_data_ner_format("/home/s4616573/data/i2b2/test.txt")
# update_CUI_all_infos("/home/s4616573/pubmed/pubmed_distill/all_CUIs.pck",'1bd1f2c2-fbab-4d54-aed9-fb442791d49d',600000,"_part_final")

def read_i2b2_file(input_file,is_i2b2):

        """Read a BIO data!"""
        rf = open(input_file, 'r')
        lines = [];
        words = [];
        labels = []
        text_lines =[]
        label_lines = []

        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if is_i2b2:
                if len(line.strip()) == 0 :#and words[-1] == '.'
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((l, w))
                    text_lines.append(w)
                    words = []
                    labels = []
            else:
                if len(line.strip()) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((l, w))
                    text_lines.append(w)
                    words = []
                    labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return text_lines,label_lines

# convert_data_ner_format("/home/s4616573/data/CLEF/train.txt","/home/s4616573/data/CLEF/","CLEF",False)
def read_xlsx_cause_effect_alps3(file_name):
    xl_file = pd.ExcelFile(file_name)

    # dfs = {sheet_name: xl_file.parse(sheet_name)
    #        for sheet_name in xl_file.sheet_names}
    dfs=xl_file.parse("Sheet1")

    causes=[[re.sub('\s+','',y) for y in x.split('\n')if len(y.strip())>0] if type(x)==str else "" for x in dfs["Root cause（Extract Target）"]]
    effects=[[re.sub('\s+','',y) for y in x.split('\n')if len(y.strip())>0] if type(x)==str else "" for x in dfs["Effects（Extract Target）"] ]
    texts=[re.sub('\s+','',x) for x in dfs["Description（Target Sentence）"]]
    causes1=dfs["Cause (new)"]

    effects1=dfs["Effect (new)"]
    print(len(texts),len(causes),len(effects),len(effects1),len(causes1))
    for i in range(227):
        if type(causes1[i]) ==str:
            sentence_causes=[re.sub('\s+','',x) for x in causes1[i].split('\n')]
            causes[i]=sentence_causes
            for x in sentence_causes:
                if x not in texts[i]:
                    print(i, "ERROR", "cause:", x, "text:", texts[i])
        if type(effects1[i]) == str:
            sentence_effects = [re.sub('\s+','',x) for x in effects1[i].split('\n')]
            effects[i] = sentence_effects
            for x in sentence_effects:
                if x not in texts[i]:
                    print(i,"ERROR","effect:",x,"text:",texts[i])

    alps3_data={"texts":texts,"causes":causes,"effects":effects}
    file_util.dump(alps3_data,"/home/s4616573/data/alps3/alps3.pck")
    ner_train_file = open("/home/s4616573/data/alps3/train.txt", "w")
    ner_test_file = open("/home/s4616573/data/alps3/test.txt", "w")
    all_labels=[]
    for t,text in enumerate(texts):
        labels=['O']*len(text)
        for effect in effects[t]:
            if len(effect.strip())>0:
                p = re.compile(effect)
                for m in p.finditer(text):
                    matched_end_index=m.end()
                    matched_start_index=m.start()
                    print(len(text),m.start(), m.group(),m.end(),effect,':',text)
                    print(text[m.start():m.end()])
                    labels[matched_start_index:matched_end_index]=["I-effect"]*(matched_end_index-matched_start_index)
                    labels[matched_start_index] = "B-effect"
                    if '。' in text[matched_start_index:matched_end_index]:
                        print(labels[matched_start_index:matched_end_index])
        for cause in causes[t]:
            if len(cause.strip())>0:
                p = re.compile(cause)
                for m in p.finditer(text):
                    matched_end_index = m.end()
                    matched_start_index = m.start()
                    print(len(text),':',m.start(),':', m.group(),':',m.end(),':',cause,':',text)


                    labels[matched_start_index:matched_end_index] = ["I-cause"] * (matched_end_index-matched_start_index)
                    labels[matched_start_index] = "B-cause"
                    if '。' in text[matched_start_index:matched_end_index]:
                        print(labels[matched_start_index:matched_end_index])
        all_labels.append(labels)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(texts, all_labels, test_size=0.2)
    for i, x in enumerate(X_train):
        for j,char in enumerate(x):
            if '。' == char and y_train[i][j] != 'O':
                print()
            ner_train_file.write(char + ' ' + y_train[i][j] + '\n')
        ner_train_file.write('\n')
    for i, x in enumerate(X_test):
        for j, char in enumerate(x):
            if '。' == char and y_test[i][j]!='O':
                print()
            ner_test_file.write(char + ' ' + y_test[i][j] + '\n')
        ner_test_file.write('\n')
    ner_test_file.close()
    ner_train_file.close()
# read_xlsx_cause_effect_alps3("/home/s4616573/data/alps3/alps_3.xlsx")

def find_max_sequence(seq,is_i2b2=True):
    # seq=[5, 3, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    non_label_idx=6 if is_i2b2 else 2
    start_idx=[i for i,x in enumerate(seq) if i<len(seq)-2 and seq[i+1]==x+1 and x<non_label_idx ]
    in_labels=[x for i,x in enumerate(seq) if i in [y+1 for y in start_idx ]]

    end_idx=[0]*len(start_idx)
    for j,start_id in enumerate(start_idx):
        for i, x in enumerate(seq):
            if i>start_id and x==in_labels[j]:
                end_idx[j]=i
            elif i>start_id and x!=in_labels[j]:
                break
    max_chunk_len = [end_idx[i]-x for i, x in enumerate(start_idx)]
    if max_chunk_len:
        max_chunk_start_id=np.argmax(max_chunk_len)
    else:
        start_idx=[0]
        end_idx=[len(seq)]
        max_chunk_start_id=0
    one_word_idx=[i for i,x in enumerate(seq) if i<len(seq)-2 and seq[i+1]!=x+1  and x<non_label_idx ]
    if len(one_word_idx)>0:
        start_idx.extend(one_word_idx)
        end_idx.extend([x+1 for x in start_idx])
    # print(end_idx)
    # print(max_chunk_start_id)

    return start_idx, end_idx,max_chunk_start_id
def get_vocab_tags(tag_file):
    vocab_tags = open(tag_file).readlines()
    vocab_tags = [re.sub('\n', '', x) for x in vocab_tags]
    vocab_tags = [x for x in vocab_tags if x != 'O']
    return vocab_tags
def find_chunk_score(tags,prob_seq,logit_sequence,id_to_tag_vocab):
    '''

    Parameters
    ----------
    seq
    labels

    Returns score_dict of each tag. For ex: {"ORG":0.8,"PER":0.2}
    -------

    '''
    tags_wth_prefix=[x.split('-')[-1] for x in tags]
    tags_wth_prefix = list(set([x for x in tags_wth_prefix if x!='O']))
    tag_score_dict = {x:[] for x in tags_wth_prefix}
    tag_sequence=[id_to_tag_vocab[x] for x in logit_sequence]
    # seq=[5, 3, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    non_label_idx=len(list(id_to_tag_vocab.keys()))
    start_idx=[i for i,x in enumerate(tag_sequence) if i<len(tag_sequence)-2 and (tag_sequence[i+1].split('-')[-1]==x.split('-')[-1] and tag_sequence[i+1].split('-')[0]!=x.split('-')[0]) and x!='O']
    in_labels=[x for i,x in enumerate(tag_sequence) if i in [y+1 for y in start_idx ]]
    sequence_tag=[]
    end_idx=[0]*len(start_idx)
    chunk_dict={}

    for j,start_id in enumerate(start_idx):
        # if start_id==1:
        #     print()
        chunk_dict[start_id] = {"end_id": -1, "label": "","score":0,"len":1}
        for i, x in enumerate(tag_sequence):
            # if i==2:
            #     print()
            x_tag=x.split('-')[-1]
            x_pre=x.split('-')[0]
            begin_tag=tag_sequence[start_id].split('-')[-1]
            begin_pre=tag_sequence[start_id].split('-')[0]
            if i>start_id and x_tag==begin_tag and x_pre!=begin_pre:
                end_idx[j]=i
                chunk_dict[start_id]["end_id"]=i
                chunk_dict[start_id]["label"]=tag_sequence[i].split('-')[-1]
                chunk_dict[start_id]["score"]=np.sum([math.log1p(x) if x >0 else 0 for x  in prob_seq[start_id:i+1]])/len(prob_seq[start_id:i+1])
                chunk_dict[start_id]["len"] +=1
            elif chunk_dict[start_id]["end_id"]!=-1:
                break
    max_chunk_len = [end_idx[i]-x for i, x in enumerate(start_idx)]
    if max_chunk_len:
        max_chunk_start_id=np.argmax(max_chunk_len)
    else:
        start_idx=[0]
        end_idx=[len(tag_sequence)]
        max_chunk_start_id=0
    one_word_idx = []
    # one_word_idx=[i for i,x in enumerate(tag_sequence) if (i<len(tag_sequence) and (tag_sequence[i+1].split('-')[-1]!=x.split('-')[-1] or (tag_sequence[i+1].split('-')[-1]==x.split('-')[-1] and tag_sequence[i+1].split('-')[0]==x.split('-')[0]))) or i==len(tag_sequence)-1 and tag_sequence[] and x!='O' ]
    for i,x in enumerate(tag_sequence):
        prev_node=tag_sequence[i-1] if i>0 else None
        next_node=tag_sequence[i+1] if i<len(tag_sequence)-1 else None
        current_node_prefix=x.split('-')[0]
        current_node_tag = x.split('-')[-1]
        is_single_tag=False
        if prev_node!=None:
            prev_node_prefix=prev_node.split('-')[0]
            prev_node_tag= prev_node.split('-')[-1]
            if x!='O' and (prev_node_prefix==current_node_prefix or (prev_node_prefix!=current_node_prefix and prev_node_tag!=current_node_tag)):
                is_single_tag=True
        if next_node!=None:
            next_node_prefix=next_node.split('-')[0]
            next_node_tag= next_node.split('-')[-1]
            if x!='O' and (next_node_prefix==current_node_prefix or (next_node_prefix!=current_node_prefix and next_node_tag!=current_node_tag)):
                is_single_tag=True
        if is_single_tag:
            one_word_idx.append(i)
    if len(one_word_idx)>0:
        for one_word_id in one_word_idx:
            chunk_dict[one_word_id]={"end_id":-1,"label":tag_sequence[one_word_id].split('-')[-1],"score":math.log1p(prob_seq[one_word_id]) if prob_seq[one_word_id]>0 else 0,"len":1}
        start_idx.extend(one_word_idx)
        end_idx.extend([x+1 for x in start_idx])
    # print(end_idx)
    # print(max_chunk_start_id)
    for start_id,item in chunk_dict.items():
        tag_score_dict[item["label"]].append(item["score"]/item["len"])
    for tag,item in tag_score_dict.items():
        tag_score_dict[tag]=np.amax(item) if len(item)>0 else 0

    source_ids=[]
    destination_ids=[]

    return tag_score_dict
# prob_seq = [0.9999993, 0.9999974, 0.9999988, 0.9999958, 0.99999547, 0.99999857, 0.99999857, 0.99994564, 0.9999975, 0.99999726, 0.9999989, 0.99999905, 0.9999969, 0.9999987, 0.99999845, 0.99999547, 0.9999987, 0.9999989, 0.9999962, 0.99999917, 0.9999994, 0.99999905, 0.9999995, 0.99999833, 0.99999905, 0.99999905, 0.9999993, 0.99999917, 0.9999995, 0.9999957, 0.9999964, 0.9999987, 0.9999989, 0.99999595, 0.9999902, 0.99999917, 0.99998677, 0.99999917, 0.9999987, 0.99999654, 0.9999982, 0.99999726, 0.9999981, 0.99999714, 0.9999988, 0.9999949, 0.99999344, 0.9999982, 0.9999987, 0.9999908, 0.99998915, 0.99999774, 0.9999988, 0.99999535, 0.999987, 0.99999785, 0.99999845, 0.9999856, 0.99998987, 0.9999976, 0.9999987, 0.9999901, 0.9999994, 0.9999443]
# logit_seq=[0, 1,4, 4, 4, 3, 4, 2, 5, 5, 5, 5, 3, 4, 3, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 6]
# start_chunk_idx, end_chunk_idx,max_pos=find_max_sequence(label_seq,True)
# start_seq=start_chunk_idx[max_pos]
# end_seq=end_chunk_idx[max_pos]
# prob_max_seq=prob_seq[start_seq:end_seq]
# bald_seq=np.average(prob_max_seq)
# tag_vocab={'B-problem':0,'B-test':1,'B-treatment':2,'I-problem':3,'I-test':4,'I-treatment':5,'O':6}
# id_to_tag={0:'B-problem',1:'B-test',2:'B-treatment',3:'I-problem',4:'I-test',5:'I-treatment',6:'O'}
# chunk_dict=find_chunk_score(prob_seq,logit_seq,id_to_tag)
# print(chunk_dict)id_to_tag={0:'B-problem',1:'B-test',2:'B-treatment',3:'I-problem',4:'I-test',5:'I-treatment',6:'O'}
# chunk_dict=find_chunk_score(prob_seq,logit_seq,id_to_tag)
# print(chunk_dict)
# a=[1,4,5,6,7]
# idx=np.array([0,1])
# b=np.array(a,~idx)
# print(b)