# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""
#todo 模型的输入
from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Embedding, Input, Flatten
from tensorflow.python.keras.regularizers import l2

from .layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from .layers.utils import Hash, concat_func, Linear,add_func

DEFAULT_GROUP_NAME = "default_group"

#todo 稀疏特征
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name', 'group_name'])):
    __slots__ = ()
    #vocabulary_size 当前特征总共有多少个不同的值
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim =  6 * int(pow(vocabulary_size, 0.25))
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name and self.embedding_name == other.embedding_name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'SparseFeat:'+self.name

#todo 连续数值型特征
class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = () #__slots__ = ('name', 'age') # 用tuple定义允许绑定的属性名称

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'DenseFeat:'+self.name

#todo 可变的稀疏特征 序列特征
class VarLenSparseFeat(namedtuple('VarLenFeat',
                                  ['name', 'maxlen', 'vocabulary_size', 'embedding_dim', 'combiner', 'use_hash',
                                   'dtype','length_name' ,'weight_name', 'embedding_name', 'group_name'])):
    __slots__ = ()

    def __new__(cls, name, maxlen, vocabulary_size, embedding_dim=8, combiner="mean", use_hash=False, dtype="float32",
                length_name=None, weight_name=None, embedding_name=None, group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim =  6 * int(pow(vocabulary_size, 0.25))
        return super(VarLenSparseFeat, cls).__new__(cls, name, maxlen, vocabulary_size, embedding_dim, combiner,
                                                    use_hash, dtype, length_name,weight_name, embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'VarLenSparseFeat:'+self.name
#todo 获取特征的名称
def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))

#todo 建立输入特征 根据输入的类型 建立Input的输入
def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):#稀疏特征
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):#数值特征
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat): #序列特征
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                     dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,),name=prefix+fc.length_name,dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


#todo feature_columns是dnn的特征 list格式 ，features是字典形式的tf输入
def input_from_feature_columns(features, feature_columns, l2_reg, init_std, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    #todo 1.0挑选稀疏特征
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    #todo 2.0 创建稀疏特征的embedding矩阵
    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, init_std, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    #todo 稀疏特征的embedding向量
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)

    #todo 3.0 获取稠密特征
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    #todo 4.0 序列特征查找
    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    #todo 5.0 将序列特征进行求mean操作
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    #todo 6.0 合并稀疏特征和序列特征
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)

    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list

#todo 1.创建embedding矩阵
def create_embedding_matrix(feature_columns, l2_reg, init_std, seed, prefix="", seq_mask_zero=True):
    #todo 只对稀疏特征创建embedding矩阵
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat) , feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

#todo 2.创建稀疏特征embedding的字典
def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {feat.embedding_name:Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                          embeddings_initializer=RandomNormal(
                                                              mean=0.0, stddev=init_std, seed=seed),
                                                          embeddings_regularizer=l2(l2_reg),
                                                          name=prefix + '_emb_' + feat.embedding_name) for feat in sparse_feature_columns}

    #todo 序列的稀疏特征
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            sparse_embedding[feat.embedding_name] = Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                              embeddings_initializer=RandomNormal(
                                                                  mean=0.0, stddev=init_std, seed=seed),
                                                              embeddings_regularizer=l2(
                                                                  l2_reg),
                                                              name=prefix + '_seq_emb_' + feat.name,
                                                              mask_zero=seq_mask_zero)
    return sparse_embedding

#todo 3.embedding向量查表
def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash: #todo 这个地方没有明白是干啥
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict

#todo 4.获取稠密特征
def get_dense_input(features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list


#todo 5.序列特征的查找
def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict

#todo 6.处理序列特征
def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                #todo seq_input是embedding向量和权值相乘以后的结果
                seq_input = WeightedSequenceLayer()(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            #todo 结合后的embedding向量
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
        if to_list:
            return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(fg.vocabulary_size, mask_zero=(feat_name in mask_feat_list))(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list



#todo 7.线性模型
def get_linear_logit(features, feature_columns, units=1, use_bias=False, init_std=0.0001, seed=1024, prefix='linear',
                     l2_reg=0):
    #todo embedding特征list
    linear_emb_list = [input_from_feature_columns(features, feature_columns, l2_reg, init_std, seed,
                                                  prefix=prefix + str(i))[0] for i in range(units)]
    _, dense_input_list = input_from_feature_columns(features, feature_columns, l2_reg, init_std, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):
        #todo mode代表是只使用deep还是即使用deep和线性部分
        if len(linear_emb_list[0]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias)([sparse_input, dense_input])

        elif len(linear_emb_list[0]) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias)(dense_input)
        else:
            #raise NotImplementedError
            return add_func([])
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)


#todo 8.拼接稀疏矩阵和稠密矩阵
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError




#todo 合并字典中的embedding向量
def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c
