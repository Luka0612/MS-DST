import Levenshtein
import json
f = open("../data/db.json")
f = json.load(f)
d_key_lower = {"name": {}, "type": {}, "area": {}, "food": {}}
d_entry = set()

for domain in f:
    for entry in f[domain]:

        if "name" in entry:
            name = entry["name"].lower()
            d_key_lower["name"][name] = entry["name"]
        else:
            name = None
        if "type" in entry:
            t = entry["type"].lower()
            d_key_lower["type"][t.lower()] = entry["type"]
        else:
            t = None
        if "area" in entry:
            area = entry["area"].lower()
            d_key_lower["area"][area.lower()] = entry["area"]
        else:
            area = None
        if "food" in entry:
            food = entry["food"].lower()
            d_key_lower["food"][food.lower()] = entry["food"]
        else:
            food = None
        d_entry.add((name, area, t, food))


def one_distance(small_text, large_text):
    small_text_len = len(small_text)
    for i in [0]:
        for j in range(0, len(large_text)):
            t = large_text[j:(j + small_text_len + i)]
            if Levenshtein.distance(t, small_text) == 1:
                return True
    return False


def distance(text, texts):
    index_text = []
    # text包含
    for i in texts:
        if i in text:
            index_text.append((i, text.index(i)))
    if index_text:
        index_text = sorted(index_text, key=lambda a:a[1])
        return index_text[0][0]

    # 包含text
    for i in texts:
        if text in i:
            index_text.append((i, len(i) - len(text)))
    if index_text:
        index_text = sorted(index_text, key=lambda a:a[1])
        return index_text[0][0]

    # 编辑距离
    distance_d = {}
    for i in texts:
        ratio = Levenshtein.ratio(i, text)
        if ratio <= 0.5:
            continue
        distance_d[ratio] = i
    if distance_d:
        distance_d = sorted(distance_d.items(), key=lambda a:a[0], reverse=True)
        return distance_d[0][1]
    else:
        return None


normalize_dict = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                  "seven": "7", "eight": "8", "nine": "9"}


def _normalize_value(value):
    normalized = value.lower()

    if normalized in normalize_dict:
        normalized = normalize_dict[normalized]

    return normalized


def _match_value(ref, pred):
    ref = _normalize_value(ref)
    pred = _normalize_value(pred)

    if ref == pred:
        result = True
    else:
        result = False

    return result


def post(dst):
    # 修改
    for domain in dst:
        if "semi" not in dst[domain]:
            continue
        for key in ["name", "area", "type", "food"]:
            if key not in dst[domain]["semi"]:
                continue
            value = dst[domain]["semi"][key]
            if len(value) != 1:
                continue
            value = value[0].lower()
            if value not in d_key_lower[key]:
                keys = list(d_key_lower[key].keys())
                other_keys = ["name", "area", "type", "food"]
                other_keys.remove(key)
                for other_key in other_keys:
                    if other_key in dst[domain]["semi"] and dst[domain]["semi"][other_key][0].lower() in keys:
                        keys.remove(dst[domain]["semi"][other_key][0].lower())

                db_value = distance(value, keys)
                if db_value:
                    db_value = d_key_lower[key][db_value]
                    dst[domain]["semi"][key] = [db_value]
            else:
                # 大小写问题
                db_value = d_key_lower[key][value]
                if db_value != dst[domain]["semi"][key][0]:
                    dst[domain]["semi"][key] = [db_value]

    # 不在db里面则删除
    for domain in dst:
        if "semi" not in dst[domain]:
            continue
        for key in ["name", "area", "type", "food"]:
            if key not in dst[domain]["semi"]:
                continue

            value = dst[domain]["semi"][key]
            if len(value) != 1:
                continue
            value = value[0].lower()
            if value not in d_key_lower[key]:
                del dst[domain]["semi"][key]

    # 不匹配删除，比如name跟area不匹配
    for domain in dst:
        if "semi" not in dst[domain]:
            continue
        values = []
        value_num = 0
        if_break = False
        for index, key in enumerate(["name", "area", "type", "food"]):
            if key not in dst[domain]["semi"]:
                values.append(None)
                if key == "name":
                    if_break = True
            else:
                if len(dst[domain]["semi"][key]) != 1:
                    if_break = True
                    break
                values.append(dst[domain]["semi"][key][0].lower())
                value_num += 1
        if if_break:
            continue
        if value_num == 1:
            continue
        min_error_entry = [None, 4, 10]
        for entry in d_entry:
            entry = list(entry)
            error_num = 0
            error_score = 0
            for index, e in enumerate(values):
                if e:
                    if entry[index] is None:
                        error_num += 1
                    elif e != entry[index] and e not in entry[index] and entry[index] not in e \
                            and not _match_value(e, entry[index]):
                        error_num += 1
                    else:
                        error_score += index
                        if e != entry[index]:
                            error_score += 0.4

            if error_num < min_error_entry[1] or min_error_entry[0] is None:
                min_error_entry = [entry, error_num, error_score]
            elif error_num == min_error_entry[1]:
                if error_score < min_error_entry[2]:
                    min_error_entry = [entry, error_num, error_score]

        if min_error_entry[1] == 1:
            for index, key in enumerate(["name", "area", "type", "food"]):

                if values[index] and values[index] != min_error_entry[0][index]:
                    if key in ["area", "food", "type"]:
                        if min_error_entry[0][index] is None:
                            continue
                        dst[domain]["semi"][key] = [d_key_lower[key][min_error_entry[0][index]]]
    return dst





