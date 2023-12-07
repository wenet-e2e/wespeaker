// Copyright (c) 2023 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace wespeaker {

const char WHITESPACE[] = " \n\r\t\f\v";

void WriteToFile(const std::string& file_path,
                 const std::vector<std::vector<float>>& embs);
void ReadToFile(const std::string& file_path,
                std::vector<std::vector<float>>* embs);

// Split the string with space or tab.
void SplitString(const std::string& str, std::vector<std::string>* strs);

void SplitStringToVector(const std::string& full, const char* delim,
                         bool omit_empty_strings,
                         std::vector<std::string>* out);

#ifdef _MSC_VER
std::wstring ToWString(const std::string& str);
#endif

}  // namespace wespeaker

#endif  // UTILS_UTILS_H_
