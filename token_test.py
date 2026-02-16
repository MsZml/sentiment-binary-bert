"""
本节小结核心介绍AI模型是如何处理字符数据的
"""
from transformers import BertTokenizer

#加载字典和分词器
token = BertTokenizer.from_pretrained(r"F:\A\sentiment-binary-bert\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
# print(token)


#准备要编码的文本数据
sents = ["白日依山尽，",
         "价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所"]

#批量编码句子
out = token.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0],sents[1]],
    add_special_tokens=True,
    #当句子长度大于max_length(上限是model_max_length)时，截断
    truncation=True,
    max_length=15,
    #一律补0到max_length
    padding="max_length",
    #可取值为tf,pt,np,默认为list
    return_tensors=None,
    return_attention_mask=True,
    return_token_type_ids=True,
    return_special_tokens_mask=True,
    #返回序列长度
    return_length=True
)
#input_ids 就是编码后的词
#token_type_ids第一个句子和特殊符号的位置是0，第二个句子的位置1（）只针对于上下文编码
#special_tokens_mask 特殊符号的位置是1，其他位置是0
#length 编码之后的序列长度
for k,v in out.items():
    print(k,":",v)

#解码文本数据
print(token.decode(out["input_ids"][0]),token.decode(out["input_ids"][1]))