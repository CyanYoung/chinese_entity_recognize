## Chinese Entity Recognize 2018-8

#### 1.preprocess

merge() 保存训练数据中的实体、jieba.load_userdict() 保证已有实体不被切分

generate() 根据 template 采样实体进行填充、生成数据，可省去、替换

check() 对分词后的数据进行标注，每句处理为 (word, pos, label) 的三元组

#### 2.represent

sent2feat() 为词增加是否句首、句尾，是否包含姓氏、数字等特征

#### 3.build

通过 crf 构建实体识别模型，使用 L1 和 L2 正则化、随机搜索超参数

#### 4.recognize

predict() 进行分词、获取词性，每句返回 (word, pred) 的二元组

#### 5.interface

response() 返回 json 字符串，每次启动生成新的日志文件、以时间命名
