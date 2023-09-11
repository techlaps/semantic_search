from transformers import T5Tokenizer
from nltk.tokenize import PunktSentenceTokenizer

#from sentence_transformers import SentenceTransformer
#model = SentenceTransformer("sentence-transformers/sentence-t5-base")
#model.save("models/sentence-t5-base")

from sentence_transformers import SentenceTransformer

from annoy import AnnoyIndex

#tokenizer = T5Tokenizer("sentence-t5-base")
tokenizer = T5Tokenizer.from_pretrained("models/sentence-t5-base")

text = "I live in New York. " + \
"Karthik planning to visit USA. " + \
"He hates the New York city. " + \
"New York is a beautiful city. Karthik lives in Chennai, India."

tokens = tokenizer.tokenize(text)
print(tokens)
print(len(tokens))


tokenizer = PunktSentenceTokenizer()
sentence_list = tokenizer.tokenize(text)
print(sentence_list)
print(len(sentence_list))

model = SentenceTransformer('models/sentence-t5-base')
embeddings = model.encode(sentence_list)
#print(embeddings)
print(len(embeddings))

#print(embeddings[0])
print(len(embeddings[0]))

f = 768  # Length of item vector(individual vector length) that will be indexed

t = AnnoyIndex(f, 'angular')
for i in range(4):
    v = embeddings[i]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

query = "Karthik is not planning to visit USA"
query_v = model.encode([query])
#print(query_v)

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
#print(u.get_nns_by_item(3, 2, include_distances=True)) # will find the n no. of nearest neighbors

print(dir(u))
print(u.get_nns_by_vector(query_v[0], 2, include_distances=True))

#print(u.get_distance(2, 3))

print(u.get_n_trees())
