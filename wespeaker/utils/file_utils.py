#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

def read_scp(scp_file):
    key_value_list = []
    with open(scp_file, "r") as fin:
        line = fin.readline()
        while line:
            tokens = line.strip().split()
            key = tokens[0]
            value = " ".join(tokens[1:])
            key_value_list.append((key, value))
            line = fin.readline()
    return key_value_list

def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists
