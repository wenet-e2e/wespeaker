# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
