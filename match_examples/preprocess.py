import random
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#todo 生成数据集,进行负采样
def gen_data_set(data, negsample=0):

    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        #todo 每个用户分组 然后计算每个用户的电影和评分
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()
        #todo 负采样
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set,size=int(len(pos_list)*negsample),replace=True)
        #todo
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1,len(hist[::-1]),rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1])))
            else:#todo 最后一个
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1]),rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]),len(test_set[0]))

    return train_set,test_set
#todo 生成模型的输入
def gen_model_input(train_set,user_profile,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])#当前评论的movid
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])
    #todo 在后面填充到具体的长度
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}
    #todo 添加用户画像的信息
    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input,train_label
