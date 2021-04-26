import os
import pickle
import json
def write_file(output_file,lines):
    f=open(output_file,"w")
    f.writelines(lines)
    f.close()
def dump(obj, save_path):
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # path = dir_path + "/" + path
    # print("PATH", save_path)
    if os.path.exists(save_path):
        # print("REMOVE PATH",save_path)
        os.remove(save_path)
    output = open(save_path, 'wb')
    pickle.dump(obj, output,protocol=4)#, protocol=pickle.HIGHEST_PROTOCOL
    output.close()
    print("SUCCESSFULLY SAVING DATA", save_path)
    # print("type", type(data))
    # datai = pickle.dumps(obj)
    # seq = datai
    # length = int(len(datai) / 10)
    # a = [seq[i:i + length] for i in range(0, len(seq), length)]
    # for i, ai in enumerate(a):
    #     output = open(save_path + str(i), 'wb')
    #     pickle.dump(ai, output)
    #     output.close()

def load(save_path):
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # path = dir_path + "/" + path
    if not os.path.exists(save_path):
        return None
    with open(save_path, 'rb') as file:
        return pickle.load(file)

def dump_json(obj,save_path):
    print("dump "+save_path)
    with open(save_path, 'w') as outfile:
        json.dump(obj, outfile)

def load_json(file_path):
    file=open(file_path,"r")
    return json.load(file)
import functools
samples=["hello1","hello2"]
import numpy as np
def get_samples(samples=["hello1","hello2"],idx=[0,1]):
    return np.array(samples)
# functools.partial(get_samples, shuffle_and_repeat=True)

