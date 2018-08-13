## Chinese Entity Recognize 2018-8

#### 1.preprocess

prepare() 将按类文件保存的训练、测试数据汇总并去重

训练数据去除停用词，统一替换地区、时间等特殊词

#### 2.represent

link_fit() 建立 class2word 字典，实现类、字、句索引的两层映射

vec_fit() 训练各类的 tfidf 模型，建立 ind2vec 字典，实现句索引、句向量的映射

#### 3.match

predict() 测试数据删除停用词，统一替换地区、时间等特殊词

通过 class2word 按字查找共现的句索引，包括同音字、同义字匹配

edit_predict() 定义编辑距离系数 edit_dist / len(phon)，将字转换为拼音

cos_predict() 余弦相关系数，通过各类的 tfidf 模型得到多个句向量