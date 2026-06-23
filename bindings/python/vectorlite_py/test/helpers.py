import sqlite3
import numpy as np
import vectorlite_py

SEED = 12345
ELEMENT_TYPES = ['float32', 'bfloat16', 'float16']
# '' (empty space) is treated as 'l2' by vectorlite.
SPACES = ['l2', 'ip', 'cosine', '']
# Reading a quantized vector back as float32 is lossy; float32 is exact.
DEQUANT_RTOL = {'float32': 0.0, 'bfloat16': 1e-2, 'float16': 1e-3}


def get_connection(path=':memory:'):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_py.vectorlite_path())
    return conn


def random_vectors(rng, n, dim):
    return np.float32(rng.random((n, dim)))


def l2_squared(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.dot(d, d))


def ip_distance(a, b):
    return float(1.0 - np.dot(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def cosine_distance(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 'l2' uses squared L2 because hnswlib does not take the square root.
DISTANCE_FN = {'l2': l2_squared, 'ip': ip_distance, 'cosine': cosine_distance}


def brute_force_knn(vectors, query, k, space='l2'):
    fn = DISTANCE_FN[space]
    dists = [(i, fn(query, vectors[i])) for i in range(len(vectors))]
    dists.sort(key=lambda x: x[1])
    return dists[:k]
