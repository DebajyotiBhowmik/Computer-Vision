#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 00:23:52 2017

@author: debajyoti
"""

import numpy as np
from sklearn.decomposition import IncrementalPCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
ipca = IncrementalPCA(n_components=2, batch_size=3)
ipca.fit(X)

trans=ipca.transform([2,3])
