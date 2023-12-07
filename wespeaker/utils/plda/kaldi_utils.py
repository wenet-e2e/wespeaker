#!/usr/bin/env python

# Copyright 2019 Lukas Burget (burget@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct

import numpy as np
from kaldi_io import open_or_fd, BadSampleSize, UnknownMatrixHeader
from kaldi_io.kaldi_io import _read_compressed_mat, _read_mat_ascii


def read_plda(file_or_fd):
    """ Loads PLDA from a file in kaldi format (binary or text).
    Input:
        file_or_fd - file name or file handle with kaldi PLDA model.
    Output:
        Tuple (mu, tr, psi) define a PLDA model using the kaldi parametrization
        mu : mean vector
        tr : transform whitening within- and diagonalizing across-class
             covariance matrix
        psi - diagonal of the across-class covariance in the transformed space
    """
    fd = open_or_fd(file_or_fd)
    try:
        binary = fd.read(2)
        if binary == b'\x00B':
            assert (fd.read(7) == b'<Plda> ')
            plda_mean = _read_vec_binary(fd)
            plda_trans = _read_mat_binary(fd)
            plda_psi = _read_vec_binary(fd)
        else:
            assert (binary + fd.read(5) == b'<Plda> ')
            plda_mean = np.array(fd.readline().strip(' \n[]').split(),
                                 dtype=float)
            assert (fd.read(2) == b' [')
            plda_trans = _read_mat_ascii(fd)
            plda_psi = np.array(fd.readline().strip(' \n[]').split(),
                                dtype=float)
        assert (fd.read(8) == b'</Plda> ')
    finally:
        if fd is not file_or_fd:
            fd.close()
    return plda_mean, plda_trans, plda_psi


def _read_vec_binary(fd):
    # Data type,
    type = fd.read(3)
    if type == b'FV ':
        sample_size = 4  # floats
    elif type == b'DV ':
        sample_size = 8  # doubles
    else:
        raise BadSampleSize
    assert (sample_size > 0)
    # Dimension,
    assert fd.read(1) == b'\4'  # int-size
    vec_size = struct.unpack('<i', fd.read(4))[0]  # vector dim
    # Read whole vector,
    buf = fd.read(vec_size * sample_size)
    if sample_size == 4:
        ans = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8:
        ans = np.frombuffer(buf, dtype='float64')
    else:
        raise BadSampleSize
    return ans


def _read_mat_binary(fd):
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    if header.startswith('CM'):
        return _read_compressed_mat(fd, header)
    elif header.startswith('SM'):
        return _read_sparse_mat(fd, header)
    elif header == 'FM ':
        sample_size = 4  # floats
    elif header == 'DM ':
        sample_size = 8  # doubles
    else:
        raise UnknownMatrixHeader("The header contained '%s'" % header)
    assert (sample_size > 0)
    # Dimensions
    s1, rows, s2, cols = \
        np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4:
        vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8:
        vec = np.frombuffer(buf, dtype='float64')
    else:
        raise BadSampleSize
    mat = np.reshape(vec, (rows, cols))
    return mat


def _read_sparse_mat(fd, format):
    """ Read a sparse matrix,
    """
    from scipy.sparse import csr_matrix
    assert (format == 'SM ')

    # Mapping for matrix elements,
    def read_sparse_vector(fd):
        _format = fd.read(3).decode()
        assert (_format == 'SV ')
        _, dim = np.frombuffer(fd.read(5), dtype='int8,int32', count=1)[0]
        _, num_elems = np.frombuffer(fd.read(5), dtype='int8,int32',
                                     count=1)[0]
        col = []
        data = []
        for j in range(num_elems):
            size = np.frombuffer(fd.read(1), dtype='int8', count=1)[0]
            dtype = 'int32' if size == 4 else 'int64'
            c = np.frombuffer(fd.read(size), dtype=dtype, count=1)[0]
            size = np.frombuffer(fd.read(1), dtype='int8', count=1)[0]
            dtype = 'float32' if size == 4 else 'float64'
            d = np.frombuffer(fd.read(size), dtype=dtype, count=1)[0]
            col.append(c)
            data.append(d)
        return col, data, dim

    _, num_rows = np.frombuffer(fd.read(5), dtype='int8,int32', count=1)[0]

    rows = []
    cols = []
    all_data = []
    max_dim = 0
    for i in range(num_rows):
        col, data, dim = read_sparse_vector(fd)
        rows += [i] * len(col)
        cols += col
        all_data += data
        max_dim = max(dim, max_dim)
    sparse_mat = csr_matrix((all_data, (rows, cols)),
                            shape=(num_rows, max_dim))
    return sparse_mat
