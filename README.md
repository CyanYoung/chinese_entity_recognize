## Chinese Entity Recognize 2018-8

#### 1.preprocess

prepare() 将按类文件保存的训练和测试数据汇总、合并

save_entity() 提取训练数据中的实体，add_entity() 添加自定义的实体

jieba.load_userdict() 保证已有实体不被切分

每句处理为 (word, pos, label) 的三元组，保存为 json 格式

#### 2.represent

sent2feat() 为词增加句首、句尾，是否包含姓氏、数字等特征

#### 3.build

CRF() 构建实体识别模型，使用 L1 和 L2 正则化

#### 4.recognize

predict() 对每句返回 (word, pred) 的二元组

#### 5.web_interface

init_entity() 按预定的 slot 对实体字典初始化

map_slot() 通过关键字将标准类映射为业务槽，适应需求变化

response() 将非空槽组合成 intent，与 entity 共同返回

每次启动生成新的日志文件，以时间命名，避免被新上传的程序覆盖

#### 6.start_service

通过用户名确定服务启动方式，先结束运行的进程，不产生 nohup.out 文件