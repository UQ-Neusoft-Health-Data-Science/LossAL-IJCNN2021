def update_dict(idx_map,new_idx):
    idx_map[idx_map["count"]]=new_idx
    idx_map["count"]=idx_map["count"]+1
def match_idx_with_root_data(top_k_indices,idx_map):
    # print ("idx data",top_k_indices)
    # print("idx_map",idx_map)
    return [idx_map[x] for x in top_k_indices]
def map_apart_of_data(all_data,part_of_idx):
    idx_map = {"count": 0}
    '''GENERATE ARRAY FROM IDX 0 FOR ANOTHER PART OF DATA'''
    [update_dict(idx_map, i) for i, x in enumerate(all_data) if i not in part_of_idx]
    return idx_map

all_data=["hello","iam fine","thank you", "and you","i love you","lovely"]
part_of_idx=[0,2]
root_map_idx=map_apart_of_data(all_data,part_of_idx)
print(map_apart_of_data(all_data,part_of_idx))
print(match_idx_with_root_data([0,1,2,3],root_map_idx))
import numpy as np
a=np.sum([0.19, 0.77, 0.96, 1.92, 3.84, 3.84, 3.84, 3.84, 9.61, 9.6, 9.6, 9.6, 19.21, 19.2])
print( a)