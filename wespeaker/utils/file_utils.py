#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang


def read_scp(scp_file):
    """ scp_file: mostly 2 columns
    """
    key_value_list = []
    with open(scp_file, "r", encoding='utf8') as fin:
        for line in fin:
            tokens = line.strip().split()
            key = tokens[0]
            value = " ".join(tokens[1:])
            key_value_list.append((key, value))
    return key_value_list


def read_lists(list_file):
    """ list_file: only 1 column
    """
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_table(table_file):
    """ table_file: any columns
    """
    table_list = []
    with open(table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            tokens = line.strip().split()
            table_list.append(tokens)
    return table_list
