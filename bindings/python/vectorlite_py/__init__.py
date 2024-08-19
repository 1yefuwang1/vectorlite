import os

__version__ = '0.2.0'

def vectorlite_path():
    loadable_path = os.path.join(os.path.dirname(__file__), 'vectorlite')
    return os.path.normpath(loadable_path)


def load_vectorlite(conn) -> None:
    conn.load_extension(vectorlite_path())
