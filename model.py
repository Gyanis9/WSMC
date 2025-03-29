import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel, BertConfig
import json
import os


class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()


class Bert_Encoder(base_model):
    """
    BERT Encoder Class用于生成句子或实体级别的表示。
    """

    def __init__(self, config, attention_probs_dropout_prob=None, hidden_dropout_prob=None, drop_out=None):
        """
        初始化BERT编码器。

        Args:
            config: 配置文件，包含模型的各种超参数，如BERT模型路径、模式等。
            attention_probs_dropout_prob: 可选，BERT中的attention层的dropout概率。
            hidden_dropout_prob: 可选，BERT中的隐藏层dropout概率。
            drop_out: 可选，Dropout层的丢弃率。
        """
        super(Bert_Encoder, self).__init__()

        # 加载预训练的BERT模型
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # 如果传入了dropout相关的参数，更新BERT配置
        if attention_probs_dropout_prob is not None:
            assert hidden_dropout_prob is not None and drop_out is not None
            self.bert_config.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.bert_config.hidden_dropout_prob = hidden_dropout_prob
            config.drop_out = drop_out

        # 定义最终输出维度（BERT的隐藏状态维度，通常是768）
        self.output_size = 768

        # Dropout层，用于防止过拟合
        self.drop = nn.Dropout(config.drop_out)

        # 根据模式选择BERT的编码策略
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')

        # 如果是 'entity_marker' 模式，调整BERT的词汇表大小，包含额外的实体标记
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + 4)  # 增加 4 个额外的标记（E11, E21, E12, E22）
            # 定义线性变换层，将BERT的输出映射到指定的维度
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 4, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        # 层归一化，用于规范化输入层
        self.layer_normalization = nn.LayerNorm([self.output_size])

    def get_output_size(self):
        """
        返回输出的维度（隐藏层的大小）。
        """
        return self.output_size

    def forward(self, inputs):
        """
        前向传播函数，用于生成句子或实体级别的表示。

        Args:
            inputs: 输入的token序列，形状为 [B, N]，其中B是batch size，N是序列长度。

        Return:
            output: 处理后的表示，形状为 [B, H*2] 或 [B, H]，取决于使用的编码策略。
        """
        # 根据编码模式生成不同的表示
        if self.pattern == 'standard':
            # 在标准模式下，使用 [CLS] 标记的表示作为句子的表示
            output = self.encoder(inputs)[1]  # [B, N] --> [B, H]
        else:
            # 在 entity_marker 模式下，使用 [E11] 和 [E21] 标记的表示作为头实体和尾实体的表示
            e11 = []  # 第一个实体的开始
            e12 = []  # 第一个实体的结束
            e21 = []  # 第二个实体的开始
            e22 = []  # 第二个实体的结束
            # 遍历batch中的每个样本
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()  # 获取 token 序列
                e11.append(np.argwhere(tokens == 30522)[0][0])  # 查找 [E11] 的位置
                e12.append(np.argwhere(tokens == 30523)[0][0])  # 查找 [E12] 的位置
                e21.append(np.argwhere(tokens == 30524)[0][0])  # 查找 [E21] 的位置
                e22.append(np.argwhere(tokens == 30525)[0][0])  # 查找 [E22] 的位置

            # 输入到BERT模型，计算出每个token的表示
            attention_mask = inputs != 0  # 注意力掩码，非零的token才计算
            tokens_output = self.encoder(inputs, attention_mask=attention_mask)[0]  # [B, N] --> [B, N, H]

            # 获取每个样本中 [E11] [E21]和 E[21] E[22]  的表示
            output = []

            for i in range(len(e11)):
                # 获取当前样本（第i个样本）的 [E11] 和 [E12] 的表示
                instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())  # 选择第i个样本
                # 获取实体的表示，拼接 [E11] 和 [E12]，以及 [E21] 和 [E22]
                entity1_rep = torch.index_select(instance_output, 1,
                                                 torch.tensor([e11[i], e12[i]]).cuda())  # [E11] 和 [E12] 的表示
                entity2_rep = torch.index_select(instance_output, 1,
                                                 torch.tensor([e21[i], e22[i]]).cuda())  # [E21] 和 [E22] 的表示

                # 拼接两个实体的表示
                entity_rep = torch.cat([entity1_rep, entity2_rep], dim=1)  # 拼接，形状为 [4, H]
                output.append(entity_rep)  # 将每个实体的表示存入列表,output 的形状是 [B, 4, H]

            # 拼接每个样本的 [E11], [E12], [E21], [E22] 的表示，得到形状为 [B*N, H*4] 的张量
            output = torch.cat(output, dim=0)  # 拼接，形状应该是 [B, 4, H]
            output = output.view(output.size()[0], -1)  # 调整形状为 [B, 4*H]
            # 将拼接后的表示输入到 Dropout 层、线性变换层和 GELU 激活函数中
            # output = self.drop(output)  # 应用 Dropout
            output = self.linear_transform(output)  # 线性变换，维度从 4*H 映射到 output_size
            output = F.gelu(output)  # GELU 激活函数
            output = self.layer_normalization(output)  # 层归一化

        return output


class proto_softmax_layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, id2rel, drop=0, config=None):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(proto_softmax_layer, self).__init__()
        self.config = config
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias=False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel
        self.rel2id = {}
        for id, rel in enumerate(id2rel):
            self.rel2id[rel] = id

    def __distance__(self, rep, rel):
        rep_ = rep.view(rep.shape[0], 1, rep.shape[-1])
        rel_ = rel.view(1, -1, rel.shape[-1])
        dis = (rep_ * rel_).sum(-1)
        return dis

    def incremental_learning(self, old_class, add_class):
        weight = self.fc.weight.data
        self.fc = nn.Linear(768, old_class + add_class, bias=False).cuda()
        with torch.no_grad():
            self.fc.weight.data[:old_class] = weight[:old_class]

    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().cuda()

    def get_feature(self, sentences):
        rep = self.sentence_encoder(sentences)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis

    def mem_forward(self, rep):
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem

    def generate_negative_samples(self, batch_representations):
        batch_size, feature_dim = batch_representations.shape
        negatives = []
        # 对每个样本生成负样本
        for i in range(batch_size):
            # 选择除当前样本外的其他所有样本
            selection = torch.cat([batch_representations[:i], batch_representations[i + 1:]])
            # 随机选择num_negatives个样本作为负样本
            indices = torch.randperm(selection.size(0))[:self.config.num_negatives]
            negative_samples = selection[indices]
            negatives.append(negative_samples)
        negatives = torch.stack(negatives, dim=0)
        return negatives

    def forward(self, sentences):
        rep = self.sentence_encoder(sentences)  # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep)
        return logits, rep


class Experience:
    def __init__(self, config):
        self.top_k = config.top_k
        self.experiences = []
        self.usage_count = []

    def add(self, sentence, labels, outputs):
        loss = F.cross_entropy(outputs, labels)
        self.experiences.append((sentence, labels, outputs, loss.item()))
        self.usage_count.append(0)

    def get_high_quality_experiences(self):
        # 选择前top_k比例的高质量经验，这里使用损失作为经验质量的衡量标准
        num_high_quality = int(len(self.experiences) * self.top_k)
        sorted_experiences = sorted(self.experiences, key=lambda x: x[3])
        high_quality_experiences = sorted_experiences[:num_high_quality]
        return high_quality_experiences

    def increment_usage(self, index):
        self.usage_count[index] += 1


def eliminate_experiences(config, experiences):
    # 获取高质量经验
    threshold = config.threshold
    high_quality_experiences = experiences.get_high_quality_experiences()

    # 使用频率计算
    usage_threshold = int(len(experiences.experiences) * threshold)
    frequent_experiences_indices = sorted(range(len(experiences.usage_count)), key=lambda x: experiences.usage_count[x],
                                          reverse=True)[:usage_threshold]

    # 获取频繁使用的经验
    frequent_experiences = [experiences.experiences[i] for i in frequent_experiences_indices]

    # 合并高质量经验和频繁使用经验，并去重
    combined_experiences = high_quality_experiences + frequent_experiences
    unique_experiences = list(set(combined_experiences))

    # 更新经验池
    experiences.experiences = unique_experiences
    experiences.usage_count = [0] * len(unique_experiences)

    return experiences
