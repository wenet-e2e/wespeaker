# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
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

import os

import fire
import numpy as np


def main(input1=None, input2=None, output=None):
    with open(input1, 'r') as file1, open(input2, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
    
    
    if len(lines1) != len(lines2):
        print("Mismatch with samples")
        return
    
    new_lines = []
    for line1, line2 in zip(lines1, lines2):
        parts1 = line1.strip().split()
        parts2 = line2.strip().split()
        
        value1 = float(parts1[2])
        value2 = float(parts2[2])
        
        average = (value1 + value2) / 2
        
        new_line = ' '.join(parts1[:2] + [f"{average:.5f}"] + parts1[3:])
        new_lines.append(new_line)
    
    with open(output, 'w') as output_file:
        for line in new_lines:
            output_file.write(line + '\n')
    
    print(f"Finishing Out_path is: {output}")



if __name__ == "__main__":
    fire.Fire(main)
