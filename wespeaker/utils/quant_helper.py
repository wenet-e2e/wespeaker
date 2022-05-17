# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re

logger = logging.getLogger(__name__)
name_width = 50  # max width of layer names
qname_width = 70  # max width of quantizer names

def set_quantizer(name, mod, quantizer, k, v):
    """Set attributes for mod.quantizer."""

    quantizer_mod = getattr(mod, quantizer, None)
    if quantizer_mod is not None:
        assert hasattr(quantizer_mod, k)
        setattr(quantizer_mod, k, v)
    else:
        logger.warn(f'{name} has no {quantizer}')

def set_quantizers(name, mod, which='both', **kwargs):
    """Set quantizer attributes for mod."""

    s = f'Warning: changing {which} quantizers of {name:{qname_width}}'
    for k, v in kwargs.items():
        s += (f' {k}={v}')
        if which in ['input', 'both']:
            set_quantizer(name, mod, '_input_quantizer', k, v)
        if which in ['weight', 'both']:
            set_quantizer(name, mod, '_weight_quantizer', k, v)
    logger.info(s)

def set_quantizer_by_name(model, names, **kwargs):
    """Set quantizer attributes for layers where name 
    contains a substring in names."""

    for name, mod in model.named_modules():
        if hasattr(mod, '_input_quantizer') or \
           hasattr(mod, '_weight_quantizer'):
            for n in names:
                if re.search(n, name):
                    set_quantizers(name, mod, **kwargs)
        elif name.endswith('_quantizer'):
            for n in names:
                if re.search(n, name):
                    s = f'Warning: changing {name:{name_width}}'
                    for k, v in kwargs.items():
                        s += (f' {k}={v}')
                        setattr(mod, k, v)
                    logger.info(s)
