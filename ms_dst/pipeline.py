import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from transformers import T5Tokenizer, T5ForConditionalGeneration
from string import ascii_letters, digits

tokenizer = T5Tokenizer.from_pretrained('t5-3b')

# 导入domain识别模型
print("export domain")
domain_model = T5ForConditionalGeneration.from_pretrained('domain_model')
domain_model.to("cuda:0")

# 导入dst识别模型
print("export dst")
dst_model = T5ForConditionalGeneration.from_pretrained('dst_model')
dst_model.to("cuda:1")

import os
import json

f = open("../data/db.json")
f = json.load(f)
db = {}
for domain in f:
    if domain not in db:
        db[domain] = {}
    for entry in f[domain]:
        for key in entry:
            if isinstance(entry[key], int):
                continue
            if not entry[key].strip():
                continue
            if key == "phone":
                continue
            if key == "id":
                continue

            db[domain][entry[key].lower()] = key

slot_keys = [
    ("restaurant", "book", "people"),
    ("restaurant", "book", "day"),
    ("restaurant", "book", "time"),
    ("restaurant", "semi", "food"),
    ("restaurant", "semi", "pricerange"),
    ("restaurant", "semi", "name"),
    ("restaurant", "semi", "area"),
    ("hotel", "book", "people"),
    ("hotel", "book", "rooms"),
    ("hotel", "book", "day"),
    ("hotel", "book", "stay"),
    ("hotel", "semi", "name"),
    ("hotel", "semi", "area"),
    ("hotel", "semi", "pricerange"),
    ("hotel", "semi", "stars"),
    ("hotel", "semi", "type"),
    ("attraction", "semi", "type"),
    ("attraction", "semi", "name"),
    ("attraction", "semi", "area")
]

domain_slot_keys = {}
for d, t, s in slot_keys:
    if d not in domain_slot_keys:
        domain_slot_keys[d] = []
    domain_slot_keys[d].append(s)

for d in domain_slot_keys:
    domain_slot_keys[d].sort()

slot_keys_type = {}
for domain, t, slot in slot_keys:
    slot_keys_type[(domain, slot)] = t


class DatasetWalker(object):
    def __init__(self, dataset, dataroot, labels=False, labels_file=None):
        path = os.path.join(os.path.abspath(dataroot))

        if dataset not in ['train', 'val', 'test']:
            raise ValueError('Wrong dataset name: %s' % (dataset))

        logs_file = os.path.join(path, dataset, 'logs.json')
        with open(logs_file, 'r') as f:
            self.logs = json.load(f)

        self.labels = None

        if labels is True:
            if labels_file is None:
                labels_file = os.path.join(path, dataset, 'labels.json')

            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

    def __iter__(self):
        if self.labels is not None:
            for log, label in zip(self.logs, self.labels):
                yield (log, label)
        else:
            for log in self.logs:
                yield (log, None)

    def __len__(self, ):
        return len(self.logs)


def get_domains(text):
    text = text.strip() + " We can know that its domains have <extra_id_0>."
    # 预测domain
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda:0")
    outputs = domain_model.generate(input_ids)
    # 后处理domains
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    domains = outputs[0].strip().split(",")
    domains.sort()
    return domains


def get_addtype_dst(text, domain):
    dbs = db[domain]
    add_type = {}
    has_add = set()
    for entry, type in dbs.items():
        if entry not in text:
            continue
        for i in range(len(text)):
            for j in [-1, 0, 1]:
                if entry != text[i: i + len(entry) - j]:
                    continue

                if has_add & set(list(range(i, i + len(entry) - j))):
                    continue
                if (i + len(entry) - j) >= len(text) or text[i + len(entry) - j] in ascii_letters + digits:
                    continue
                if (i - 1) >= 0 and text[i - 1] in ascii_letters + digits:
                    continue
                other_text = text[i: i + len(entry) - j]
                if entry != other_text:
                    continue
                add_type[(i, i + len(entry) - j)] = (entry, type)
                has_add = has_add | set(list(range(i, i + len(entry) - j)))

    add_type_sort = sorted(add_type.items(), key=lambda x: x[0][1], reverse=True)

    for index, type in add_type_sort:
        text = text[: index[1]] + "(" + type[1] + ")" + text[index[1]:]

    slot_type_text = ""
    slots = domain_slot_keys[domain]
    for i in range(len(slots)):
        slot_type_text += "%s is <extra_id_%s> , " % (slots[i], str(i))
    slot_type_text = slot_type_text[:-2] + "."
    text = text + "From domain %s, we can know that " % domain + slot_type_text
    print("input text:", text)

    # 模型预测
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda:1")
    print("input_ids:", input_ids.shape)
    outputs = dst_model.generate(input_ids, max_length=64)
    print("outputs:", outputs.shape)

    # 后处理
    pred = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True)[0]
    print("slot_type_text:", slot_type_text)
    print("pred:", pred)

    _, domain_and_slot = text.rsplit("From domain", 1)

    slot_index_dict = {}
    slot_and_indexs = domain_and_slot.strip().split("we can know that")[1].strip().split(",")
    for slot_and_index in slot_and_indexs:
        slot_and_index = slot_and_index.split("is")
        slot_index_dict[slot_and_index[1].strip().split("<extra_id_")[-1].split(">")[0]] = slot_and_index[0].strip()

    slots = pred.replace("<pad>", "").replace("</s>", "").strip().split("<extra_id_")
    dst_result = {}
    for slot_value in slots:
        if not slot_value.strip():
            continue
        slot_value = slot_value.split(">", 1)
        if len(slot_value) != 2:
            continue
        if slot_value[0] not in slot_index_dict:
            continue
        slot = slot_index_dict[slot_value[0]]
        slot_value[1] = slot_value[1].strip()
        if slot_value[1] == "None":
            continue

        slot_value_text = slot_value[1]
        values = slot_value_text.split(",")
        if (domain, slot) not in slot_keys_type:
            continue
        slot_type = slot_keys_type[(domain, slot)]

        if slot_type not in dst_result:
            dst_result[slot_type] = {}
        dst_result[slot_type][slot] = values
    return dst_result


from post import post


def pipeline(text):
    domains = get_domains(text)
    dst = {}
    for domain in domains:
        if domain not in ["restaurant", "hotel", "attraction"]:
            continue
        dst_result = get_addtype_dst(text, domain)
        dst[domain] = dst_result

    print("not post:", dst)
    dst = post(dst)
    print("post:", dst)
    return dst


def walker():
    data = DatasetWalker("test", "../data/", labels=False)

    dsts = []
    for log, label in data:
        text = ""
        for utr in log:
            text += utr["speaker"] + ":" + utr["text"] + " "

        uttr = text.strip() + ". "
        uttr = uttr.replace("U:", "user:").replace("S:", "system:")

        print("uttr:", uttr)
        dst = pipeline(uttr)
        print("dst:", dst)
        print()
        dsts.append(dst)

    w = open("./labels.json", "w")
    json.dump(dsts, w)


if __name__ == '__main__':
    walker()
