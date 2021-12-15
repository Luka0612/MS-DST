from dataset_walker import DatasetWalker

import sys
import json
import argparse

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

normalize_dict = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                  "seven": "7", "eight": "8", "nine": "9"}

class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self._total_num_instances = 0.0
        self._joint_goal_matched = 0.0

        self._total_num_slots = 0.0
        self._num_slots_matched = 0.0

        self._ref_slots_with_values = 0.0
        self._ref_slots_with_none = 0.0

        self._pred_slots_with_values = 0.0
        self._pred_slots_with_none = 0.0

        self._value_match_score = 0.0
        self._none_match_score = 0.0

    def _normalize_value(self, value):
        normalized = value.lower()

        if normalized in normalize_dict:
            normalized = normalize_dict[normalized]

        return normalized

    def _match_value(self, ref, pred):
        ref = self._normalize_value(ref)
        pred = self._normalize_value(pred)

        if ref == pred:
            result = True
        else:
            result = False

        return result
                    
    def update(self, ref_obj, pred_obj):
        joint_goal_flag = True
        
        for key1, key2, key3 in slot_keys:
            self._total_num_slots += 1
            
            if key1 in ref_obj and key2 in ref_obj[key1] and key3 in ref_obj[key1][key2]:
                ref_val = list(set(ref_obj[key1][key2][key3]))
            else:
                ref_val = None

            if key1 in pred_obj and key2 in pred_obj[key1] and key3 in pred_obj[key1][key2]:
                pred_val = list(set(pred_obj[key1][key2][key3]))
            else:
                pred_val = None

            if ref_val is None and pred_val is None:
                self._ref_slots_with_none += 1
                self._pred_slots_with_none += 1

                self._none_match_score += 1
                self._num_slots_matched += 1
            elif ref_val is None and pred_val is not None:
                self._ref_slots_with_none += 1
                self._pred_slots_with_values += 1
                
                joint_goal_flag = False
            elif ref_val is not None and pred_val is None:
                self._ref_slots_with_values += 1
                self._pred_slots_with_none += 1
                
                joint_goal_flag = False                
            else:
                self._ref_slots_with_values += 1
                self._pred_slots_with_values += 1
                
                num_matched_values = 0.0
                for r in ref_val:
                    for p in pred_val:
                        if self._match_value(r, p):
                            num_matched_values += 1

                if num_matched_values > 0.0:
                    prec_values = num_matched_values/len(pred_val)
                    rec_values = num_matched_values/len(ref_val)
                    f1_values = 2*prec_values*rec_values/(prec_values+rec_values)
                else:
                    f1_values = 0.0

                self._value_match_score += f1_values
                
                if f1_values == 1.0:
                    self._num_slots_matched += 1
                else:
                    joint_goal_flag = False

        if joint_goal_flag is True:
            self._joint_goal_matched += 1

        self._total_num_instances += 1
        
    def scores(self):
        jga = self._joint_goal_matched / self._total_num_instances

        slot_accuracy = self._num_slots_matched / self._total_num_slots

        if self._pred_slots_with_values > 0:
            slot_value_p = self._value_match_score / self._pred_slots_with_values
        else:
            slot_value_p = 0.0

        if self._ref_slots_with_values > 0:
            slot_value_r = self._value_match_score / self._ref_slots_with_values
        else:
            slot_value_r = 0.0

        if (slot_value_p + slot_value_r) > 0.0:
            slot_value_f = 2 * slot_value_p * slot_value_r / (slot_value_p + slot_value_r)
        else:
            slot_value_f = 0.0

        if self._pred_slots_with_none > 0:
            slot_none_p = self._none_match_score / self._pred_slots_with_none
        else:
            slot_none_p = 0.0

        if self._ref_slots_with_none > 0:
            slot_none_r = self._none_match_score / self._ref_slots_with_none
        else:
            slot_none_r = 0.0

        if (slot_none_p + slot_none_r) > 0.0:
            slot_none_f = 2 * slot_none_p * slot_none_r / (slot_none_p + slot_none_r)
        else:
            slot_none_f = 0.0

        scores = {
            'joint_goal_accuracy': jga,
            'slot': {
                'accuracy': slot_accuracy,
                'value_prediction': {
                    'prec': slot_value_p,
                    'rec': slot_value_r,
                    'f1': slot_value_f
                },
                'none_prediction': {
                    'prec': slot_none_p,
                    'rec': slot_none_r,
                    'f1': slot_none_f
                }
            }
        }

        return scores
        
def main(argv):
    parser = argparse.ArgumentParser(description='Evaluate the system outputs.')

    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', choices=['train', 'val', 'test'], required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store', metavar='PATH', required=True,
                        help='Will look for corpus in <dataroot>/<dataset>/...')
    parser.add_argument('--outfile',dest='outfile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing output JSON')
    parser.add_argument('--scorefile',dest='scorefile',action='store',metavar='JSON_FILE',required=True,
                        help='File containing scores')

    args = parser.parse_args()

    with open(args.outfile, 'r') as f:
        output = json.load(f)
    
    data = DatasetWalker(dataroot=args.dataroot, dataset=args.dataset, labels=True)

    metric = Metric()

    for (instance, ref), pred in zip(data, output):
        metric.update(ref, pred)
        
    scores = metric.scores()

    with open(args.scorefile, 'w') as out:
        json.dump(scores, out, indent=2)
    

if __name__ =="__main__":
    main(sys.argv)        
