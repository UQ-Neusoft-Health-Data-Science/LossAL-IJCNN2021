import os
import operator
import re
import nltk
import copy
import utils.ai_const as sony_const
from nltk.corpus import words as nltk_words
import numpy as np
import utils.file_util as file_util
import math
from sentence_transformers import SentenceTransformer, util
from transformers import BertConfig, BertModel,BertTokenizer
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
# from pattern.text.en import singularize
# def convert_plural_to_singular(text):
#     return singularize(text)
def clean_text(text):
    text = re.sub("'","",text)
    text=re.sub(("(\\W)+"," ",text))
    return text
def find_whole_word(w,line):
    regex_pt=re.compile(r'\b({0})\b'.format(re.escape(w)))
    results=regex_pt.search(line)
    if not results:
        return None
    return results
def get_range_words(text):
    return re.findall("(?:minimum|maximum|offset|ratio|axis|width|length)\s+",text)
def get_negative_words(text):
    words = get_pattern_from_text_nltk("JJ",text)
    negative_words=[]
    for word in words:
        positions, match_groups= match_with_position("^un|^non|^de|^dis|^anti|^im|^il|^ir|^a",word)
        if len(positions)>0:
            begin_indices=[i for i,x in enumerate(positions) if x==0]
            match_group=match_groups[begin_indices[0]]
            root_index=len(match_group)
            if root_index>0:
                root_word=word[root_index:]
                if len(root_word)>1:
                    if root_word in nltk_words.words():
                        negative_words.append(word)
    return negative_words

def remove_characters(removed_regex,regex_converted_to, input_text):
    return re.sub(removed_regex,regex_converted_to,input_text)

def nltk_tokenize(text):
    pos_tag_list = nltk.pos_tag(text.split())
    words = [x[0] for x in pos_tag_list]
    tags = [x[1] for x in pos_tag_list]
    return words, tags

def match_with_position(pattern, sentence):
    p = re.compile(pattern)
    positions = []
    matched_groups = []
    for m in p.finditer(sentence):
        # print(m.start(), m.group())
        positions.append(m.start())
        # print(sentence[0:m.start()-1])
        matched_groups.append(m.group())
    return positions, matched_groups

def get_position_from_pos_tag(tag_pos, tag_phrases, words, tag_str):
    token_count = len(tag_phrases.split('#'))
    token_count = token_count - 1 if tag_phrases.endswith("#") else token_count
    consider_text = tag_str[0:tag_pos + 1]
    tokens_before = len(consider_text.split('#'))
    tokens_before = 0 if tokens_before == 1 or (tokens_before == 2 and consider_text.endswith('#')) else tokens_before
    tokens_before = tokens_before - 1 if tag_str[0:tag_pos].endswith("#") else tokens_before
    related_words = words[tokens_before:tokens_before + token_count]
    related_words = [re.sub('<.*>|”|“', '', x) for x in related_words]
    related_words = [re.sub('-', ' ', x) for x in related_words]
    name_mention = " ".join(related_words)
    return name_mention

def not_special_chars_inside(text):
    return len(re.findall("^[_A-z]*((-|\s)*[_A-z])*$", text)) > 0

def get_pattern_from_text_nltk(pattern, text):
    words, tags = nltk_tokenize(str(text))
    words = [re.sub('<.*>|”|“', '', x) for x in words]
    pos_punct = ["VBD" if x in sony_const.noun_tag_list and words[i] in sony_const.include_verbs else x for i, x in enumerate(tags)]
    pos_punct = ["CC" if (x in sony_const.noun_tag_list or x in sony_const.other_tag_list) and words[i] in sony_const.conjunctions else x for i, x in
                 enumerate(pos_punct)]
    pos_punct = ["UNK" if (x in sony_const.other_tag_list or x in sony_const.noun_tag_list) and words[i] in sony_const.no_meaning_words else x for i, x
                 in enumerate(pos_punct)]
    pos_punct = ["UNK" if not not_special_chars_inside(words[i]) or len(words[i]) == 1 else x for i, x in
                 enumerate(pos_punct)]
    post_punct_str = "#".join(pos_punct)
    tag_pos, matched_tag_groups = match_with_position(pattern, post_punct_str)
    matched_texts = []
    str_to_match = post_punct_str
    for j, matched in enumerate(matched_tag_groups):
        # print("MATCH",matched_tag_groups[j],tag_pos[j],str_to_match[tag_pos[j]])
        matched_text = get_position_from_pos_tag(tag_pos[j], matched, words, str_to_match)
        other_words = re.findall('\(.*\)', matched_text)
        other_words = [re.sub(sony_const.brackets, '', x) for x in other_words]
        if len(other_words) > 0:
            matched_text = re.sub("|".join(other_words), '', matched_text)
            matched_texts.extend(other_words)
        matched_text = re.sub(sony_const.brackets, '', matched_text)
        number_words = ['^' + x + '\s+' for x in sony_const.number_words]
        matched_text = re.sub('|'.join(number_words), '', matched_text)
        matched_text = re.sub('^(?:\w\s+)+|(?:\s+\w)+$', '', matched_text)
        matched_text = re.sub('\s{2,}', ' ', matched_text)
        matched_texts.append(matched_text)
    matched_texts = [x.strip() for x in matched_texts]
    return matched_texts

def get_name_mention_from_claims_nltk(claims):
    name_mention_pattern = "(?:JJ#|NN#|NNS#|NNP#|NNPS#)*(?:NN|NNS|NNP|NNPS)+"
    name_mentions = []
    for i, claim in enumerate(claims):
        short_texts = re.split(',|;|:|\.', str(claim))
        for short_text in short_texts:
            targets = get_pattern_from_text_nltk(name_mention_pattern, short_text)
            # print(claim)
            name_mentions.extend(targets)
        # print("claim:", i)
    name_mentions = [x for x in name_mentions if
                     len(x.split()) >= 2 or (len(x.split()) == 1 and not_special_chars_inside(x) and len(x) > 1)]
    name_mentions = list(set(name_mentions))
    return name_mentions

def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    # Replace all none alphanumeric characters with spaces
    # s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def generate_n_gram_from_name_mentions(name_mentions,n_gram_range):
    all_name_mention=[]
    for name_mention in name_mentions:
        for i in range(n_gram_range):
            ngram_count=i+1
            ngram_list=generate_ngrams(name_mention,ngram_count)
            all_name_mention.extend(ngram_list)
            all_name_mention.append(name_mention)
    all_name_mention=list(set(all_name_mention))
    return all_name_mention
text="nonplanar defect amorphous an antifuse varactor formed on the substrate structure, the antifuse varactor having a third gate terminal"
# print(get_negative_words(text))
# print(get_range_words(text))name_mentions = text_util.get_name_mention_from_claims_nltk(claims)
#     n_graprint len(re.findall(r'\w+', line))print len(re.findall(r'\w+', line))m_name_mentions = text_util.generate_n_gram_from_name_mentions(name_mentions, max_n_gram_range)
def remove_html_tag(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

def count_token(lines):
    # count_arr=[len(re.findall(r'\w+', x)) for x in lines]
    count_arr =[len(x.split()) for x in lines]

    return  np.sum(count_arr)
# lines=open("/home/s4616573/data/uts/cui_dict_umls_siteCLEF.json").readlines()
# for i,line in enumerate(lines):
#     lines[i]=remove_html_tag(line)
# print(count_token(["hello em yeu ."]))
# consider_sorted_idx=[100,200,300]
# prob_mapping={i : w for i, w in enumerate(consider_sorted_idx)}
# print(prob_mapping)

def match_idx_with_root_data(top_k_indices,idx_map):
    print ("idx data",top_k_indices)
    print("idx_map",idx_map)
    return [idx_map[x] for x in top_k_indices]
def sort_score_by_len(best_probs,word_len_list,sorted_len_idx,current_token_sum,token_total,all_idx,start_idx,step=10):
    # sorted_len_idx = np.argsort([x for x in word_len_list])[::-1]
    consider_sorted_idx=sorted_len_idx[start_idx:start_idx+step]
    consider_probs=best_probs[consider_sorted_idx]
    prob_mapping={i : w for i, w in enumerate(consider_sorted_idx)}
    len_sorted_prob_idx=consider_probs.argsort()[-len(consider_sorted_idx):][::-1]
    top_10_idx=match_idx_with_root_data(len_sorted_prob_idx,prob_mapping)
    for idx in top_10_idx:
        if current_token_sum<token_total:
            all_idx.append(idx)
            current_token_sum+=word_len_list[idx]
        else:
            break
    print(current_token_sum)
    if current_token_sum<token_total:
        sort_score_by_len(best_probs, word_len_list, sorted_len_idx,  current_token_sum, token_total,all_idx, start_idx+step)
def create_simi_matrix(sentences=[],dissi_neighbor_num=10, simi_neighbor_num=10):
    model = SentenceTransformer('average_word_embeddings_glove.840B.300d')
    if len(sentences)==0:
        sentences = ['The cat sits outside',
                 'A man is playing guitar',
                 'I love pasta',
                 'The new movie is awesome',
                 'The cat plays in the garden',
                 'A woman watches TV',
                 'The new movie is so great',
                 'Do you like pizza?']


    #Compute embeddings
    embeddings = model.encode(sentences)#, convert_to_tensor=True

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores =np.array(util.pytorch_cos_sim(embeddings, embeddings))#file_util.load("simi_scores.pck")#
    diss_cosine_scores=copy.deepcopy(cosine_scores)
    cosine_scores_full = copy.deepcopy(cosine_scores)
    # limited_neighbor_num = int(len(cosine_scores[0])*0.3)
    # file_util.dump(cosine_scores,"simi_scores.pck")
    dissi_dict={}
    simi_dict={}
    for i,x in enumerate(cosine_scores):
        simi_neighbor_i = np.zeros(x.shape)
        dissi_neighbor_i= np.zeros(x.shape)
        idx_sort_asc=x.argsort()[:len(x)]
        idx_sort_asc=[y for y in idx_sort_asc if not math.isnan(x[y])]
        cosine_scores_i_sortedup = idx_sort_asc[0:dissi_neighbor_num + 1]

        cosine_scores_i_sorteddown=idx_sort_asc[len(cosine_scores_i_sortedup)-simi_neighbor_num-1:]
        simi_neighbor_i[cosine_scores_i_sorteddown]=x[cosine_scores_i_sorteddown]
        dissi_neighbor_i[cosine_scores_i_sortedup] = 1-x[cosine_scores_i_sortedup]
        cosine_scores[i]=simi_neighbor_i
        diss_cosine_scores[i]=dissi_neighbor_i
        dissi_dict[i]=cosine_scores_i_sortedup
        simi_dict[i]=cosine_scores_i_sorteddown
        # x_=np.zeros(x.shape)
        # c=x[cosine_scores_i_sorteddown]
        # a=simi_neighbor_i[np.nonzero(simi_neighbor_i)]
        # b=dissi_neighbor_i[np.nonzero(dissi_neighbor_i)]
        # np.where(x,x)
        # cosine_scores[i]=[y if j !=i and j in  cosine_scores_i_sorteddown else 0 for j,y in enumerate(x) ]
        print("cosine score",i)
        # diss_cosine_scores[i]=[1-y if j !=i and j in  cosine_scores_i_sortedup else 0 for j,y in enumerate(x) ]

    len_sentences=len(sentences)
    # matrix_all1=np.ones(len_sentences*len_sentences).reshape((len_sentences, len_sentences))
    # cosine_scores=[[1-y for y in x] for x in cosine_scores]
    # for i in range(len(cosine_scores) - 1):
    #     for j in range(i + 1, len(cosine_scores)):
    #         cosine_scores[i][j]=1-cosine_scores[i][j]
    # print(np.array(cosine_scores))
    #Find the pairs with the highest cosine similarity scores
    # pairs = []
    # for i in range(len(cosine_scores)-1):
    #     for j in range(i+1, len(cosine_scores)):
    #         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
    #
    # #Sort scores in decreasing order
    # pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    #
    # for pair in pairs[0:10]:
    #     i, j = pair['index']
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
    # if converted2csc:
    #     return convert2csc_matrix(cosine_scores)
    del model
    del embeddings
    return cosine_scores_full,diss_cosine_scores,dissi_dict,simi_dict


def create_simi_matrix_train_test(sentences=[],test_sentences=[],neighbor_num=100,using_bert=False):
    desired_thres_dict={}
    desired_thres_list=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
    if len(sentences)==0:
        sentences = ['The cat sits outside',
                 'A man is playing guitar',
                 'I love pasta',
                 'The new movie is awesome',
                 'The cat plays in the garden',
                 'A woman watches TV',
                 'The new movie is so great',
                 'Do you like pizza?']
    if using_bert:
        from transformers import AutoTokenizer, AutoModel

        tokenizer =SentenceTransformer('stsb-roberta-base')
        # model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

        test_embedding = tokenizer.encode(test_sentences)
        embeddings = tokenizer.encode(sentences)  # , convert_to_tensor=True
    else:
        model = SentenceTransformer('average_word_embeddings_glove.840B.300d')
        embeddings = model.encode(sentences)  # , convert_to_tensor=True
        test_embedding = model.encode(test_sentences)



    #Compute embeddings

    #Compute cosine-similarities for each sentence with each other sentence
    train_cosine_scores =np.array(util.pytorch_cos_sim(embeddings, embeddings))#file_util.load("simi_scores.pck")#
    test_cosine_scores=np.array(util.pytorch_cos_sim(embeddings, test_embedding))
    # limited_neighbor_num = int(len(cosine_scores[0])*0.3)
    # file_util.dump(cosine_scores,"simi_scores.pck")
    dissi_dict={}
    simi_dict={}

    len_sentences=len(sentences)
    # matrix_all1=np.ones(len_sentences*len_sentences).reshape((len_sentences, len_sentences))
    # cosine_scores=[[1-y for y in x] for x in cosine_scores]
    # for i in range(len(cosine_scores) - 1):
    #     for j in range(i + 1, len(cosine_scores)):
    #         cosine_scores[i][j]=1-cosine_scores[i][j]
    # print(np.array(cosine_scores))
    #Find the pairs with the highest cosine similarity scores
    # pairs = []
    # for i in range(len(cosine_scores)-1):
    #     for j in range(i+1, len(cosine_scores)):
    #         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
    #
    # #Sort scores in decreasing order
    # pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    #
    # for pair in pairs[0:10]:
    #     i, j = pair['index']
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
    # if converted2csc:
    #     return convert2csc_matrix(cosine_scores)
    test_sim_diss_vector=np.zeros((len(sentences),neighbor_num*2))
    sim_diss_idx=[]
    idx_sort_asc_list = [[y  for y in x.argsort()[:len(x)] ] for x in test_cosine_scores]
    for id,simi_threshold in enumerate(desired_thres_list):
        upper_value=desired_thres_list[id-1] if id>0 else -10
        # if upper_value!=-10:
        #     desired_thres_dict[simi_threshold]={u_target_id: np.where((x >= simi_threshold)&(x<upper_value))[0]  for u_target_id,x in enumerate(test_cosine_scores)}
        # else:
        #     desired_thres_dict[simi_threshold] = {u_target_id: np.where(x >= simi_threshold)[0]
        #                                           for u_target_id, x in enumerate(test_cosine_scores)}
        desired_thres_dict[simi_threshold] = {u_target_id: np.where(x >= simi_threshold)[0]
                                              for u_target_id, x in enumerate(test_cosine_scores)}
    # for i,x in enumerate(test_cosine_scores):
    #     # idx_sort_asc = x.argsort()[:len(x)]
    #     # idx_sort_asc = [y for y in idx_sort_asc if not math.isnan(x[y])]
    #     # idx_sort_asc=[y for y in idx_sort_asc_list[i] if not math.isnan(x[y])]
    #     simi_train_test_over_05[i]= np.where(test_cosine_scores[i] >= 0.5)[0]
        # print("i:",i)
        # if len(idx_sort_asc)>0:
        #     dissi_idx=idx_sort_asc[:neighbor_num]
        #     test_sim_diss_vector[i][neighbor_num:neighbor_num*2]=np.array(x)[dissi_idx]#dissi_array
        #     simi_idx= idx_sort_asc[len(idx_sort_asc)-neighbor_num:len(idx_sort_asc)]
        #     test_sim_diss_vector[i][0:neighbor_num] =np.array(x)[simi_idx]
        #     sim_diss_idx.append((dissi_idx,simi_idx))
        # else:
        #     # test_sim_diss_vector[i][neighbor_num:neighbor_num * 2] = np.zeros((neighbor_num * 2))  # dissi_array
        #
        #     # test_sim_diss_vector[i][0:neighbor_num*2] =np.zeros((neighbor_num * 2))
        #     sim_diss_idx.append(([], []))

            # a=test_sim_diss_vector[i][0:100]
    return train_cosine_scores,test_cosine_scores,desired_thres_dict
# dict1={'hello':1,'a':2,'c':3}
# print({value:key for key,value in dict1.items()})
# size_analysis=100
# best_probs=np.random.random(size_analysis)
# word_len_list=np.array([x for x in range(size_analysis)])
# print("probs",best_probs)
# print("word_len",word_len_list)
# sorted_len_idx = np.argsort([x for x in word_len_list])[::-1]
# token_limit=300
# current_token_sum=0
# top_token_idx=[]
# sort_score_by_len(best_probs, word_len_list, sorted_len_idx, current_token_sum, token_limit, top_token_idx, 0,3)
# print( "selected idx",top_token_idx)
# print("sum",np.sum(word_len_list[top_token_idx]))
# probs=[[-0.0019977,-0.00200119] ,[-0.00199997, -0.00195026], [0.0200059, -0.00199843]]
# import math
# prob_seq = [ math.log1p(float(x)) if x!=0 else 0  for x in np.amax(probs, axis=1)]
# [7, 9 ,8, 6, 5 ,4, 3 ,2, 1 ,0] {0: 1379, 1: 3354, 2: 2910, 3: 2372, 4: 180, 5: 5297, 6: 6345, 7: 5946, 8: 1193}
# import math
# print(math.log1p(-1.1238033))
def read_data(input_file):
        """Read a BIO data!"""

        def process_line(labels, words):
            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            lines.append((l, w))
            words = []
            labels = []
            return words, labels, lines

        rf = open(input_file, 'r')
        lines = [];
        words = [];
        labels = []
        for line in rf:
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check

            if len(line.strip()) == 0:  # and words[-1] == '.'
                words, labels, lines = process_line(labels, words)
            words.append(word)
            labels.append(label)
        rf.close()
        return lines

def create_simi_matrix_full(sentences=[],dissi_neighbor_num=10, simi_neighbor_num=10):
    model = SentenceTransformer('average_word_embeddings_glove.840B.300d')
    if len(sentences)==0:
        sentences = ['The cat sits outside',
                 'A man is playing guitar',
                 'I love pasta',
                 'The new movie is awesome',
                 'The cat plays in the garden',
                 'A woman watches TV',
                 'The new movie is so great',
                 'Do you like pizza?']


    #Compute embeddings
    embeddings = model.encode(sentences)#, convert_to_tensor=True

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores =np.array(util.pytorch_cos_sim(embeddings, embeddings))#file_util.load("simi_scores.pck")#
    cosine_scores_full=copy.deepcopy(cosine_scores)
    diss_cosine_scores=copy.deepcopy(cosine_scores)

    # limited_neighbor_num = int(len(cosine_scores[0])*0.3)
    # file_util.dump(cosine_scores,"simi_scores.pck")
    dissi_dict={}
    simi_dict={}
    for i,x in enumerate(cosine_scores):
        simi_neighbor_i = np.zeros(x.shape)
        dissi_neighbor_i= np.zeros(x.shape)
        idx_sort_asc=x.argsort()[:len(x)]
        idx_sort_asc=[y for y in idx_sort_asc if not math.isnan(x[y])]
        cosine_scores_i_sortedup = idx_sort_asc[0:dissi_neighbor_num + 1]

        cosine_scores_i_sorteddown=idx_sort_asc[len(cosine_scores_i_sortedup)-simi_neighbor_num-1:]
        simi_neighbor_i[cosine_scores_i_sorteddown]=x[cosine_scores_i_sorteddown]
        dissi_neighbor_i[cosine_scores_i_sortedup] = 1-x[cosine_scores_i_sortedup]
        cosine_scores[i]=simi_neighbor_i
        diss_cosine_scores[i]=dissi_neighbor_i
        dissi_dict[i]=cosine_scores_i_sortedup
        simi_dict[i]=cosine_scores_i_sorteddown
        # x_=np.zeros(x.shape)
        # c=x[cosine_scores_i_sorteddown]
        # a=simi_neighbor_i[np.nonzero(simi_neighbor_i)]
        # b=dissi_neighbor_i[np.nonzero(dissi_neighbor_i)]
        # np.where(x,x)
        # cosine_scores[i]=[y if j !=i and j in  cosine_scores_i_sorteddown else 0 for j,y in enumerate(x) ]
        print("cosine score",i)
        # diss_cosine_scores[i]=[1-y if j !=i and j in  cosine_scores_i_sortedup else 0 for j,y in enumerate(x) ]

    len_sentences=len(sentences)
    # matrix_all1=np.ones(len_sentences*len_sentences).reshape((len_sentences, len_sentences))
    # cosine_scores=[[1-y for y in x] for x in cosine_scores]
    # for i in range(len(cosine_scores) - 1):
    #     for j in range(i + 1, len(cosine_scores)):
    #         cosine_scores[i][j]=1-cosine_scores[i][j]
    # print(np.array(cosine_scores))
    #Find the pairs with the highest cosine similarity scores
    # pairs = []
    # for i in range(len(cosine_scores)-1):
    #     for j in range(i+1, len(cosine_scores)):
    #         pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
    #
    # #Sort scores in decreasing order
    # pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    #
    # for pair in pairs[0:10]:
    #     i, j = pair['index']
    #     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
    # if converted2csc:
    #     return convert2csc_matrix(cosine_scores)
    del model
    del embeddings
    return cosine_scores_full,diss_cosine_scores,dissi_dict,simi_dict
# dataset_dir="/home/s4616573/data/i2b2/"
# lines = read_data(os.path.join(dataset_dir,"train.txt"))
# all_texts_data = np.array([x[1] for x in lines])
# create_simi_matrix(all_texts_data)

# trained_matrix= np.zeros(shape=(9,9))
# dissi_dict={0:[1,2,3,4],1:[7,8,9],2:[6]}
# trained_indices=[0,2]
# for tid in trained_indices:
#     rows=[tid]*len(dissi_dict[tid])
#     trained_matrix[rows, dissi_dict[tid]] = 1
# print()
def get_simi_dissi_vector(target_idx, simi_scores,trained_indices,neighbor_num):
    sim_diss_vector=np.zeros((len(target_idx),neighbor_num*2))
    simi_dissi_idx=[]
    for i,target_id in enumerate(target_idx):
        x=simi_scores[target_id]
        idx_sort_asc =np.array(trained_indices)[ x[trained_indices].argsort()[:len(trained_indices)]]
        idx_sort_asc = [y for y in idx_sort_asc if not math.isnan(x[y])]
        # print("i:",i)
        if len(idx_sort_asc)>0:
            dissi_idx=idx_sort_asc[0:neighbor_num]
            sim_diss_vector[i][neighbor_num:neighbor_num*2]=np.array(x)[dissi_idx]#dissi_array
            simi_idx= idx_sort_asc[len(idx_sort_asc)-neighbor_num:len(idx_sort_asc)]
            sim_diss_vector[i][0:neighbor_num] =np.array(x)[simi_idx]
            simi_dissi_idx.append((simi_idx,dissi_idx))
        else:
            simi_dissi_idx.append(([],[]))

    return sim_diss_vector,simi_dissi_idx
def is_special_char(text):
    return re.findall("\{|\}|\[|\]|\(|\)|\d+",text)
# lines1=file_util.load("/home/s4616573/data/bert_ner_output/conll_delta_label_alpha_0_delta_all_tokens/selected_text_15.pck")
# # lines2=file_util.load("/home/s4616573/data/bert_ner_output/conll_delta_label_alpha_0_delta_all_tokens/selected_text_21.pck")
# #
# idx1=file_util.load("/home/s4616573/data/bert_ner_output/conll_delta_label_alpha_0_delta_all_tokens/selected_idx_15.pck")
# #
# for i,line in enumerate(lines1):
#     print("line1:",i,line,'-',idx1[i],len(is_special_char(line))>0)
# import collections
# print([(item,count) for item, count in collections.Counter(lines1).items() if count > 1])

# for line in lines2:
#     print("line2",line,len(is_special_char(line))>0)

# a=[1,2,3,4,5]
# print(a[-3:-1])
# import scipy
# a= np.array([ 1, 1, 0, 2, 3, 4, 5])
# # a=scipy.sparse.csr_matrix(a)
# # print(a.todense())
# print(a.argsort()[-len(a):][::-1])
# simi_test_instances={1:[1,2,3],2:[3,4,5]}
# a=[1,2,3]
# a=np.array([v for k,v in simi_test_instances.items() if k in a[0:2]]).ravel()
# print(a)
from transformers import AutoTokenizer, AutoModel
# sentences = ['The cat sits outside',
#                  'A man is playing guitar',
#                  'I love pasta',
#                  'The new movie is awesome',
#                  'The cat plays in the garden',
#                  'A woman watches TV',
#                  'The new movie is so great',
#
#                  'Do you like pizza?']
# tokenizer = st('paraphrase-distilroberta-base-v1')#"dmis-lab/biobert-base-cased-v1.1"
#     # model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
#
# embeddings = tokenizer.encode(sentences)
# print(embeddings[0])
# cosine_scores =util.pytorch_cos_sim(embeddings, embeddings)
# print(cosine_scores[2])
# a=np.array([x for x in range(10)])
# print(np.where((a>2)&(a<5)))
# b=np.where(a>2)
# for x in b[0]:
#     print(x)

# i2b2_dataset_dir="/home/s4616573/data/i2b2/"
# clef_dataset_dir="/home/s4616573/data/CLEF/"
# conll_dataset_dir = "/home/s4616573/data/conll-corpora/conll2003/"
# clef_test_lines=read_data(os.path.join(clef_dataset_dir, "test.txt"))
# conll_test_lines=read_data(os.path.join(conll_dataset_dir, "test.txt"))
# i2b2_test_lines=read_data(os.path.join(i2b2_dataset_dir, "test.txt"))
# print("lines:",len(clef_test_lines),"tokens:",count_token([x[1] for x in clef_test_lines]))
# print("lines:",len(conll_test_lines),"tokens:",count_token([x[1] for x in conll_test_lines]))
# print("lines:",len(i2b2_test_lines),"tokens:",count_token([x[1] for x in i2b2_test_lines]))
# lines: 3453 tokens: 46435
# lines: 45052 tokens: 396157
