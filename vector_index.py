import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Document:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata

    def __str__(self):
        return f"<Document text='{self.text[:10]}...' metadata={self.metadata}>"

    def __repr__(self):
        return str(self)

class Section:
    def __init__(self, text, parent_document):
        self.text = text
        self.parent_document = parent_document

    def __str__(self):
        return f"<Section text='{self.text[:10]}...' parent_document={repr(self.parent_document)}>"

    def __repr__(self):
        return str(self)


class VectorIndex:
    INDEX_SAVE_NAME = "%s/index"
    INDEX_VECS_SAVE_NAME = "%s/index_vecs"

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.all_sections = []
        self.vecs = None
    
    def build_index(self, documents):
        self.documents = documents
        vectors = []
        all_sections = []
        for doc in tqdm(documents, desc='Encoding documents'):
            sections = VectorIndex.split_doc(doc)
            section_vecs = self.model.encode([section.text for section in sections],
                                             convert_to_tensor=False)
            vectors.extend(section_vecs)
            all_sections.extend(sections)
        self.all_sections = all_sections
        self.vecs = np.vstack(vectors)
    
    def search(self, query, n=-1):
        query_vec = self.model.encode(query, convert_to_tensor=False)
        scores = np.dot(self.vecs, query_vec.T)
        best_matches_indices = np.argsort(scores)[::-1]
        results = [{'section': self.all_sections[i], 'score': scores[i]} for i in best_matches_indices]
        if n == -1:
            return results
        return results[:n]

    def split_doc(doc, chunk_size=1000, overlap=150):
        if overlap * 2 >= chunk_size:
            raise ValueError("Overlap must be less than half the chunk size.")
        chunks = []
        start = 0
        while start < len(doc.text):
            if start > 0:
                start -= overlap
            end = min(start + chunk_size, len(doc.text))
            chunks.append(Section(text=doc.text[start:end], parent_document=doc))
            start = end
        return chunks
    
    def save(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        index_fname = self.INDEX_SAVE_NAME % savedir
        index_vecs_fname = self.INDEX_VECS_SAVE_NAME % savedir
        with open(index_fname, 'wb') as f:
            data = {'documents': self.documents, 'all_sections': self.all_sections}
            pickle.dump(data, f)
        np.save(index_vecs_fname, self.vecs)
    
    @classmethod
    def load(cls, savedir):
        instance = cls()
        index_fname = VectorIndex.INDEX_SAVE_NAME % savedir
        index_vecs_fname = VectorIndex.INDEX_VECS_SAVE_NAME % savedir
        with open(index_fname, 'rb') as f:
            data = pickle.load(f)
            instance.documents = data['documents']
            instance.all_sections = data['all_sections']
        instance.vecs = np.load(f"{index_vecs_fname}.npy")
        return instance