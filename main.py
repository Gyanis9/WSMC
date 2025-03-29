import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
from copy import deepcopy
from nni.utils import merge_parameter
from tqdm import tqdm
from model import Bert_Encoder, proto_softmax_layer, Experience, eliminate_experiences
from data_loader import get_data_loader
from sampler import data_sampler
from transformers import BertTokenizer
from sklearn.metrics import f1_score, classification_report
from config import get_config
from utils import save_representation_to_file
import nni
import torch.nn.functional as F
from utils import set_seed, select_data, get_proto, get_aca_data

default_print = "\033[0m"
blue_print = "\033[1;34;40m"
yellow_print = "\033[1;33;40m"
green_print = "\033[1;32;40m"


def evaluate(config, test_data, seen_relations, rel2id, flag=None, pid2name=None, model=None):
    model.eval()
    n = len(test_data)
    data_loader = get_data_loader(config, test_data, batch_size=128)
    gold = []
    pred = []
    correct = 0
    with torch.no_grad():
        seen_relation_ids = [rel2id[rel] for rel in seen_relations]
        for _, (_, labels, sentences, _, _) in enumerate(data_loader):
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            _, rep = model(sentences)
            logits = model.get_mem_feature(rep)
            predicts = logits.max(dim=-1)[1].cpu()
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long()
            correct += (predicts == labels).sum().item()
            predicts = predicts.tolist()
            labels = labels.tolist()
            gold.extend(labels)
            pred.extend(predicts)
    micro_f1 = f1_score(gold, pred, average='micro')
    macro_f1 = f1_score(gold, pred, average='macro')
    if flag is not None:
        if len(pid2name) != 0:
            seen_relations = [x + pid2name[x][0] for x in seen_relations]
        print('\n' + classification_report(gold, pred, labels=range(len(seen_relations)), target_names=seen_relations,
                                           zero_division=0))
        print(f"Micro F1 Score: {micro_f1}")
        print(f"Macro F1 Score: {macro_f1}")
    return correct / n


def wake_phase(config, task_id, model, experiences_pool, train_set):
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])
    for epoch_i in range(config.wake_epochs):
        for step, (_, labels, sentences, _, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            logits, _ = model(sentences)
            labels = labels.cuda()
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if epoch_i == 1:
                experiences_pool.add(sentences, labels, logits)
    return model, experiences_pool


def utilize_experience(config, experiences_pool, task_id, model):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])
    for epoch_i in range(1):
        high_quality_experiences = experiences_pool.get_high_quality_experiences()
        for experience in high_quality_experiences:
            exp_sentences, labels, exp_logits, _ = experience
            model.zero_grad()
            logits, _ = model(exp_sentences)
            labels = labels.cuda()
            ce_loss = criterion(logits, labels)
            soft_log_probs = F.log_softmax(logits / config.temperature, dim=1)
            soft_targets = F.softmax(exp_logits / config.temperature, dim=1)
            KD_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean') * (config.temperature ** 2)
            CE_loss = F.cross_entropy(logits, labels)
            distillation_loss = config.alpha * KD_loss + (1 - config.alpha) * CE_loss
            loss = ce_loss + distillation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model, experiences_pool


def rem_phase(config, model, train_set, epochs):
    data_loader = get_data_loader(config, train_set, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss()
    criterion2 = Dream_Loss(config)
    optimizer = optim.Adam([
        {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])

    for epoch_i in range(epochs):
        losses = []
        for step, (_, labels, sentences, _, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            logits, _ = model(sentences)
            labels = labels.cuda()
            ce_loss = criterion(logits, labels)
            adv_data = Generate_adversarial(config, sentences)
            adv_logits, _ = model(adv_data)
            conb_loss = criterion2(adv_logits, logits, labels)
            loss = conb_loss + ce_loss
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model


def nrem_phase(config, task_id, model, mem_set, epochs, current_proto, seen_relation_ids, pre_model):
    data_loader = get_data_loader(config, mem_set, shuffle=True)

    model.train()
    if task_id > 0:
        pre_model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
        {'params': model.fc.parameters(), 'lr': 0.001}
    ])
    pre_logits = None
    for epoch_i in range(epochs):
        model.set_memorized_prototypes(current_proto)
        losses = []
        for step, (_, labels, sentences, _, _) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            cur_logits, rep = model(sentences)
            logits_proto = model.mem_forward(rep)
            if task_id > 0:
                with torch.no_grad():
                    pre_logits, _ = pre_model(sentences)
                    pre_logits = pre_logits.detach()
            labels = torch.tensor([seen_relation_ids.index(i.item()) for i in labels]).long().to(config.device)
            ce_loss = criterion(logits_proto, labels)
            negatives = model.generate_negative_samples(rep)
            Contrastive_Focal_Distillation_Loss = combined_loss(task_id, rep, rep, negatives, pre_logits, cur_logits,
                                                                labels, alpha=config.alpha,
                                                                beta=config.beta, temperature=config.temperature,
                                                                contrastive_temperature=config.contrastive_temperature)
            loss = ce_loss + Contrastive_Focal_Distillation_Loss
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    return model


def combined_loss(task_id, anchor, positive, negatives, pre_logits, cur_logits, labels,
                  alpha=0.5, beta=0.5, temperature=2.0, contrastive_temperature=0.5):
    # 计算自监督对比损失
    batch_size, num_negatives, feature_dim = negatives.shape

    # 计算anchor与positive之间的相似度
    positive_similarity = F.cosine_similarity(anchor, positive, dim=-1)
    positive_similarity = positive_similarity.unsqueeze(1)  # 增加维度以便广播

    # 计算anchor与每个负样本之间的相似度
    # anchor 需要扩展维度以匹配负样本的数量
    anchor_expanded = anchor.unsqueeze(1).expand(-1, num_negatives, -1)
    negative_similarities = F.cosine_similarity(anchor_expanded, negatives, dim=-1)

    # 使用softmax来计算概率
    all_similarities = torch.cat([positive_similarity, negative_similarities], dim=1)
    probabilities = F.log_softmax(all_similarities / contrastive_temperature, dim=1)
    # 对比损失是-log(softmax的第一个元素)
    contrastive_loss = -probabilities[:, 0].mean()
    if task_id > 0:
        # 计算焦点知识蒸馏损失
        soft_log_probs = F.log_softmax(cur_logits / temperature, dim=1)
        soft_targets = F.softmax(pre_logits / temperature, dim=1)
        KD_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean') * (temperature ** 2)
        CE_loss = F.cross_entropy(cur_logits, labels)
        distillation_loss = alpha * KD_loss + (1 - alpha) * CE_loss
    else:
        distillation_loss = 0

    # 结合两种损失
    combined_total_loss = beta * contrastive_loss + (1 - beta) * distillation_loss

    return combined_total_loss


def Generate_adversarial(config, x):
    x = x.float()
    noise = torch.randn_like(x)
    norm = torch.norm(noise, p=config.p1, dim=tuple(range(1, x.dim())), keepdim=True)
    noise = noise / norm * config.gamma
    x = x + noise
    return x.long()


class Dream_Loss(nn.Module):
    def __init__(self, config):
        super(Dream_Loss, self).__init__()
        self.p = config.p2
        self.weight1 = config.weight1
        self.weight2 = config.weight2
        self.weight3 = config.weight3

    def adversarial_robustness_loss(self, logits, y):
        return F.cross_entropy(logits, y)

    def adversarial_comp_sum_p_margin_loss(self, logits, y):
        correct_class_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        max_logit, _ = logits.max(dim=1)
        margin = 1.0 - (max_logit - correct_class_logit)
        loss = torch.clamp(margin, min=0).mean()
        return loss

    def p_margin_loss(self, logits, y):
        correct_class_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        max_logit, _ = logits.max(dim=1)
        margin = 1.0 - (max_logit - correct_class_logit) / (self.p - 1)
        loss = torch.clamp(margin, min=0).mean()
        return loss

    def forward(self, logits_adv, logits_clean, y):
        loss1 = self.adversarial_robustness_loss(logits_adv, y)
        loss2 = self.adversarial_comp_sum_p_margin_loss(logits_adv, y)
        loss3 = self.p_margin_loss(logits_clean, y)
        return self.weight1 * loss1 + self.weight2 * loss2 + self.weight3 * loss3


if __name__ == '__main__':
    config = get_config()
    tuner_params = nni.get_next_parameter()
    config = merge_parameter(config, tuner_params)
    config.exp_name = f'{config.task_name}'
    if not os.path.exists(f'reps/{config.exp_name}'):
        os.mkdir(f'reps/{config.exp_name}')

    tokenizer = BertTokenizer.from_pretrained(config.bert_path,
                                              additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    task_results = []
    memory_results = []
    pid2name = json.load(open('data/pid2name.json', 'r')) if config.task_name.lower() == 'fewrel' else {}
    s = time.time()
    for i in range(config.total_round):
        if not os.path.exists(f'reps/{config.exp_name}/{i}'):
            os.mkdir(f'reps/{config.exp_name}/{i}')

        test_acc = []
        memory_acc = []
        set_seed(config.seed + i * 100)
        sampler = data_sampler(config=config, seed=config.seed + i * 100, tokenizer=tokenizer)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        encoder = Bert_Encoder(config=config).cuda()

        add_relation_num = config.rel_per_task * 3
        model = proto_softmax_layer(encoder, num_class=len(sampler.id2rel) + add_relation_num, id2rel=sampler.id2rel,
                                    drop=0, config=config).cuda()
        experience_pool = Experience(config)
        memorized_samples = {}
        pre_model = None

        for task_id, (
                training_data, valid_data, test_data, current_relations, historic_test_data,
                seen_relations) in enumerate(
            sampler):
            print(f"{yellow_print}Training Task {task_id}: Relation Set = {current_relations}.{default_print}")
            short_term_memory = []
            for relation in current_relations:
                short_term_memory += training_data[relation]

            aug_data = get_aca_data(config, deepcopy(training_data), current_relations, tokenizer)
            print(
                f'{yellow_print}Task {task_id}: Total number of data samples = {len(short_term_memory + aug_data)}{default_print}')
            model, experience_pool = wake_phase(config, task_id, model, experience_pool, short_term_memory + aug_data)
            eliminate_experiences(config, experience_pool)
            model, experience_pool = utilize_experience(config, experience_pool, task_id, model)
            model.incremental_learning(config.num_of_relation, add_relation_num)
            # experience_pool = Experience(config)

            print(f'{blue_print}Selecting Optimal Examples for Long_Term_Memory Consolidation{default_print}')
            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, training_data[relation])
            long_term_memory = []
            for rel in memorized_samples:
                long_term_memory += memorized_samples[rel]
            dreaming_data = long_term_memory + short_term_memory
            print(
                f"{yellow_print}Engaging in NREM and REM cycle training for enhanced continual relation extraction{default_print}")
            seen_relation_ids = [rel2id[rel] for rel in seen_relations]
            for _ in range(config.sleep_epochs):
                protos4train = []
                for relation in seen_relations:
                    protos4train.append(get_proto(config, encoder, memorized_samples[relation]))
                protos4train = torch.cat(protos4train, dim=0).detach()
                model = nrem_phase(config, task_id, model, long_term_memory, 1, protos4train, seen_relation_ids,
                                   pre_model)
                model = rem_phase(config, model, dreaming_data, 1)
            protos4eval = []
            for relation in seen_relations:
                r = model.fc.weight[rel2id[relation]].detach()
                proto = get_proto(config, encoder, memorized_samples[relation], r)
                proto = proto / proto.norm()
                protos4eval.append(proto)
            protos4eval = torch.cat(protos4eval, dim=0).detach()
            model.set_memorized_prototypes(protos4eval)

            print(f'{blue_print}Evaluation...{default_print}')
            current_test_data = []
            for relation in current_relations:
                current_test_data += test_data[relation]

            all_test_data = []
            for relation in seen_relations:
                all_test_data += historic_test_data[relation]
            cur_acc = evaluate(config, current_test_data, seen_relations, rel2id, pid2name=pid2name,
                               model=model)
            total_acc = evaluate(config, all_test_data, seen_relations, rel2id, flag=True,
                                 pid2name=pid2name,
                                 model=model)
            save_representation_to_file(config, model, sampler, id2rel, f'reps/{config.exp_name}/{i}/{task_id}.pt',
                                        memorized_samples)

            print(f'{blue_print}Restart Attempt {i + 1}{default_print}')
            print(f'{blue_print}Processing Task {task_id + 1}{default_print}')
            test_acc.append(cur_acc)
            memory_acc.append(total_acc)
            print(f'{green_print}Memory Accuracy: {memory_acc}{default_print}')
            print(f'{green_print}Task Accuracy: {test_acc}{default_print}')
            pre_model = model
        task_results.append(test_acc)
        memory_results.append(memory_acc)
        average_acc = sum(memory_acc) / len(memory_acc)
        print(f"{green_print}Current Average Accuracy: {average_acc}{default_print}")
        nni.report_intermediate_result(average_acc)
    e = time.time()
    task_results = torch.tensor(task_results, dtype=torch.float32)
    memory_results = torch.tensor(memory_results, dtype=torch.float32)
    print(f"{default_print}All task results:{default_print}")
    for i in range(task_results.size(0)):
        print(task_results[i].tolist())
    print(f"{default_print}All memory results:{default_print}")
    for i in range(memory_results.size(0)):
        print(memory_results[i].tolist())
    task_results = torch.mean(task_results, dim=0).tolist()
    memory_results = torch.mean(memory_results, dim=0)
    final_average = torch.mean(memory_results).item()
    print(f"{green_print}Final task results: {task_results}")
    print(f"{green_print}Final memory results: {memory_results.tolist()}")
    print(f"Final average accuracy: {final_average}")
    print(f"Total time taken: {e - s}s")
    nni.report_final_result(final_average)
