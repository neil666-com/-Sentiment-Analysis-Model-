import torch
import argparse
import numpy as np  
from data_utils import Tokenizer4Bert
from models.bert_spc import BERT_SPC
from transformers import BertModel


MODEL_WEIGHT_PATH = 'state_dict/bert_spc_restaurant_val_acc_0.8411'  
PRETRAINED_BERT_PATH = 'D:/ABSA-PyTorch-master/bert-base-uncased'   
MAX_SEQ_LEN = 85                                                    


# 情感标签映射：0=负面，1=中性，2=正面
polarity_map = {0: '负面', 1: '中性', 2: '正面'}

def init_model(opt):
    """初始化模型"""
    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = BERT_SPC(bert, opt).to(opt.device)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=opt.device))
    model.eval()  
    return model, tokenizer

def absa_infer(sentence, aspect, model, tokenizer, opt):
    """
    ABSA推理核心函数
    :param sentence: 待分析的句子（餐厅领域，如：这家店的牛排超嫩，服务却很差）
    :param aspect: 评价方面（如：牛排、服务、环境、价格）
    :return: 情感倾向（正面/负面/中性）
    """
    # 用项目的tokenizer处理句子+方面，和训练时的输入格式保持一致
    text_bert_indices, bert_segments_ids = tokenizer.encode_bert_tokens(sentence, aspect)
    # 构造模型输入
    inputs = [
        torch.tensor(np.array([text_bert_indices]), dtype=torch.long).to(opt.device),
        torch.tensor(np.array([bert_segments_ids]), dtype=torch.long).to(opt.device)
    ]
    # 模型推理
    with torch.no_grad():
        outputs = model(inputs)
    # 取预测结果
    polarity_idx = torch.argmax(outputs, dim=-1).item()
    return polarity_map[polarity_idx]

if __name__ == '__main__':
    # 配置推理参数（和训练时的参数保持一致）
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--pretrained_bert_name', default=PRETRAINED_BERT_PATH, type=str)
    parser.add_argument('--max_seq_len', default=MAX_SEQ_LEN, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()
    # 设置设备（和训练时一致，CPU/GPU）
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    
    # 初始化模型和分词器
    model, tokenizer = init_model(opt)
    print("模型加载完成，可开始情感分析推理！")
    print("格式说明：输入句子+空格+评价方面，输入q退出")
    
    # 交互式推理（输入句子和方面，实时出结果）
    while True:
        input_str = input("\n请输入(例：这家店的牛排超嫩服务却很差 牛排(要输入英文)):")
        if input_str.lower() == 'esc':
            print("退出推理！")
            break
        try:
           
            input_str = input_str.replace('\u3000', ' ').strip()  
            if ' ' not in input_str:  
                raise ValueError("缺少分隔空格")
            sentence, aspect = input_str.split(' ', 1)  
            if not sentence or not aspect:
                raise ValueError("句子或评价方面不能为空")
            sentiment = absa_infer(sentence, aspect, model, tokenizer, opt)
            print(f"评价方面：{aspect} | 情感倾向：{sentiment}")
        except ValueError as ve:
            print("输入格式错误！请按「句子 评价方面」的格式输入，例：这家店的环境很好 环境")
        except Exception as e:
            # 其他异常（如模型推理错误）提示具体信息，方便排查
            print(f"推理出错：{str(e)}，请检查输入内容或模型配置是否正确")