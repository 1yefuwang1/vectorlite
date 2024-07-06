import vectorlite_py
import numpy as np
import apsw

"""
A contrived example of using vectorlite to search vectors with metadata filter.
Metadata filter in vectorlite is achived by filtering rowid.
Candidate rowids need to be generated first then passed to the `rowid in (..)` constraint.
The rowid constraint is pushed down to the HNSW index when traversing the HNSW graph, so it is efficient and accurate.

"""

conn = apsw.Connection(':memory:')
conn.enable_load_extension(True) # enable extension loading
conn.load_extension(vectorlite_py.vectorlite_path()) # loads vectorlite

cursor = conn.cursor()

DIM = 32 # dimension of the vectors
NUM_ELEMENTS = 1000 # number of vectors

# In this example, we have a news table that stores article.
cursor.execute(f'create table news(rowid integer primary key, article text)')
# For simplicity, the article is just a string of the rowid.
cursor.executemany('insert into news(rowid, article) values (?, ?)', [(i, str(i)) for i in range(NUM_ELEMENTS)])

# Create a virtual table to store the embeddings
cursor.execute(f'create virtual table vector_table using vectorlite(article_embedding float32[{DIM}], hnsw(max_elements={NUM_ELEMENTS}))')
# For simplicity, embeddings are randomly generated for each article.
# In a real application, you should replace this with your own embeddings.
data = np.float32(np.random.random((NUM_ELEMENTS, DIM))) # Only float32 vectors are supported by vectorlite for now
embeddings = [(i, data[i].tobytes()) for i in range(NUM_ELEMENTS)]
cursor.executemany('insert into vector_table(rowid, article_embedding) values (?, ?)', embeddings)

# Now let's search for the 10 nearest neighbors of the first article in articles that start with "1"
result = cursor.execute(f'select rowid, distance from vector_table where knn_search(article_embedding, knn_param(?, 10)) and rowid in (select rowid from news where article like "1%")', [data[0].tobytes()]).fetchall()
print(result)

# Please prefer using rowid in(...) instead of using `join`. The below query will first find 10 neighbors and then filter by rowid, which is not what we wanted.
# result = cursor.execute(f'select a.rowid, a.distance from vector_table a join news b on a.rowid = b.rowid where b.article like "1%" and knn_search(my_embedding, knn_param(?, 10)) ', [data[0].tobytes()]).fetchall()
