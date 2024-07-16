import os
import json
import ast
import re
import copy
from collections import OrderedDict

import virtualhome_eval.simulation.evolving_graph.utils as utils

class Vocabulary:
    def __init__(self, file_path, id_to_name_dict):
        self.vocab = self.load_vocab(file_path)
        self.state_param_dict = self.get_tl_predicates_param_dict()
        self.actio_param_dict = self.get_subgoal_actions()
        self.id_to_name_dict = id_to_name_dict
        
    def load_vocab(self, file_path):
        with open(file_path, 'r') as f:
            vocab = json.load(f)
        return vocab
    
    def get_vocab(self):
        return self.vocab
    
    def get_tl_predicates(self):
        return self.vocab['tl_predicates']
    
    def get_actions_all(self):
        return self.vocab['actions']
    
    def get_actions_all_in_list(self):
        return list(self.vocab['actions'].keys())
    
    def get_subgoal_actions(self):
        return self.vocab['subgoal_actions']
    
    def get_subgoal_actions_in_list(self):
        return list(self.vocab['subgoal_actions'].keys())
    
    def get_vh_info(self):
        '''
        Returns the vocabulary for properties, states and relations in VH
        '''
        return self.vocab['properties'], self.vocab['vh_states'], self.vocab['vh_relations']
    
    def get_tl_to_vh_predicates_dict(self):
        return self.vocab['tl_predicates_to_vh']
    
    def get_vh_states_to_tl_dict(self):
        return self.vocab['vh_states_to_tl']
    
    def get_vh_relations_to_tl_dict(self):
        return self.vocab['vh_relations_to_tl']
    
    def get_tl_predicates_param_dict(self):
        tl_predicates_to_vh = self.get_tl_to_vh_predicates_dict()
        vh_states_to_tl = self.get_vh_states_to_tl_dict()
        vh_relations_to_tl = self.get_vh_relations_to_tl_dict()
        vh_properties, vh_states, vh_relations = self.get_vh_info()
        param_dict = {}
        for tl_predicate in self.get_tl_predicates():
            vh_predicate = tl_predicates_to_vh[tl_predicate]
            param_num = -1
            if vh_predicate in vh_properties:
                param_num = 1
            elif vh_predicate in vh_states and vh_states_to_tl[vh_predicate] == tl_predicate:
                param_num = 1
            elif vh_predicate in vh_relations and vh_relations_to_tl[vh_predicate] == tl_predicate:
                param_num = 2
            param_dict[tl_predicate] = param_num
        return param_dict
    

valid_actions = {
    "DRINK": ("DRINK", 1),
    "EAT": ("EAT", 1),
    "CUT": ("CUT", 1),
    "TOUCH": ("TOUCH", 1),
    "LOOKAT": ("LOOKAT", 1),
    "LOOKAT_SHORT": ("LOOKAT_SHORT", 1),
    "LOOKAT_MEDIUM": ("LOOKAT_MEDIUM", 1),
    "LOOKAT_LONG": ("LOOKAT_LONG", 1),
    "WATCH": ("WATCH", 1),
    "READ": ("READ", 1),
    "TYPE": ("TYPE", 1),
    "PUSH": ("PUSH", 1),
    "PULL": ("PULL", 1),
    "MOVE": ("MOVE", 1),
    "SQUEEZE": ("SQEEZE", 1),
    "SLEEP": ("SLEEP", 0),
    "WAKEUP": ("WAKEUP", 0),
    "RINSE": ("RINSE", 1),
    "SCRUB": ("SCRUB", 1),
    "WASH": ("WASH", 1),
    "GRAB": ("GRAB", 1),
    "SWITCHOFF": ("SWITCHOFF", 1),
    "SWITCHON": ("SWITCHON", 1),
    "CLOSE": ("CLOSE", 1),
    "FIND": ("FIND", 1),
    "WALK": ("WALK", 1),
    "OPEN": ("OPEN", 1),
    "POINTAT": ("POINTAT", 1),
    "PUTBACK": ("PUTBACK", 2),
    "PUTIN": ("PUTIN", 2),
    "PUTOBJBACK": ("PUTOBJBACK", 1),
    "RUN": ("RUN", 1),
    "SIT": ("SIT", 1),
    "STANDUP": ("STANDUP", 0),
    "TURNTO": ("TURNTO", 1),
    "WIPE": ("WIPE", 1),
    "PUTON": ("PUTON", 1),
    "PUTOFF": ("PUTOFF", 1),
    "GREET": ("GREET", 1),
    "DROP": ("DROP", 1),
    "LIE": ("LIE", 1),
    "POUR": ("POUR", 2),
}

state_transform_dictionary = {
    "CLOSED": "CLOSED",
    "OPEN": "OPEN",
    "ON": "ON",
    "OFF": "OFF",
    "SITTING": "SITTING",
    "DIRTY": "DIRTY",
    "CLEAN": "CLEAN",
    "LYING": "LYING",
    "PLUGGED_IN": "PLUGGED_IN",
    "PLUGGED_OUT": "PLUGGED_OUT",
    "ONTOP": "ONTOP",  # relation on should be converted into ontop
    "OBJ_ONTOP": "OBJ_ONTOP",
    "ON_CHAR": "ON_CHAR",
    "INSIDE": "INSIDE",
    "OBJ_INSIDE": "OBJ_INSIDE",
    "INSIDE_ROOM": "INSIDE_ROOM",
    "BETWEEN": "BETWEEN",
    "NEXT_TO": "NEXT_TO",
    "OBJ_NEXT_TO": "OBJ_NEXT_TO",
    "FACING": "FACING",
    "HOLDS_RH": "HOLDS_RH",
    "HOLDS_LH": "HOLDS_LH",
    "SITTINGRELATION": "ONTOP",  # relation sitting should be converted into ontop
}


def my_scene_evaluate(
    final_state_dict,
    selected_node_goals,
    selected_edge_goals,
    character_id,
    action_seq,
    action_goals,
):
    nodes = final_state_dict["nodes"]
    edges = final_state_dict["edges"]
    node_tot_num = len(selected_node_goals)
    node_success_num = 0
    edge_tot_num = len(selected_edge_goals)
    edge_success_num = 0
    action_tot_num = len(action_goals)
    action_success_num = 0
    for node_dict in nodes:
        cur_id = node_dict["id"]
        cur_class_name = node_dict["class_name"]
        cur_states = node_dict["states"]
        for node_goal in selected_node_goals:
            goal_id = node_goal["id"]
            goal_class_name = node_goal["class_name"]
            goal_state = node_goal["state"]
            if cur_id == goal_id and cur_class_name == goal_class_name and goal_state in cur_states:
                node_success_num += 1
    for edge_dict in edges:
        cur_from_id = edge_dict["from_id"]
        cur_to_id = edge_dict["to_id"]
        cur_relation = edge_dict["relation_type"]
        for edge_goal in selected_edge_goals:
            goal_from_id = edge_goal["from_id"]
            goal_to_id = edge_goal["to_id"]
            goal_relation = edge_goal["relation_type"]
            if cur_from_id == goal_from_id and cur_to_id == goal_to_id and cur_relation == goal_relation:
                edge_success_num += 1
                break
    if len(action_goals) > 0:
        for action_goal in action_goals:
            action_candidates = action_goal.split("|")
            success = False
            for action in action_candidates:
                found = False
                for action_instruction in action_seq:
                    if action in action_instruction:
                        found = True
                        break
                if found:
                    success = True
                    action_success_num += 1
                    break
            if not success:
                break
    tot_success_num = node_success_num + edge_success_num + action_success_num
    tot_num = node_tot_num + edge_tot_num + action_tot_num
    return node_tot_num, node_success_num, edge_tot_num, edge_success_num, action_tot_num, action_success_num, tot_num, tot_success_num
