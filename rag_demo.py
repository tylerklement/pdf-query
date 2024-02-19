import argparse

from llmware.prompts import Prompt

from pdf_extractor import PDFExtractor
from vector_index import VectorIndex, Document


class RAGModel:
    def __init__(self, vector_index, model_name="llmware/bling-1b-0.1"):
        self.prompter = Prompt().load_model(model_name)
        self.vector_index = vector_index

    def query(self, query, temperature=0.0):
        retrieved_result = self.vector_index.search(query, n=1)[0]
        output = self.prompter.prompt_main(query,
                                           context=retrieved_result['section'].text,
                                           prompt_name="default_with_context",
                                           temperature=temperature)
        return {'answer': output['llm_response'].strip(), 'context': retrieved_result}


def main(pdf_dir, tesseract_path, queries):
    pdf_extractor = PDFExtractor(tesseract_path)
    pdfs = pdf_extractor.load_pdfs_as_text(pdf_dir)
    documents = [Document(text, {'filename': fname}) for fname, text in pdfs.items()]
    vector_index = VectorIndex()
    vector_index.build_index(documents)
    rag_model = RAGModel(vector_index)
    print("Documents loaded.")

    if len(queries) == 0:
        # get user input
        while True:
            query = input("Enter question -> ")
            print("Processing answer...")
            answer = rag_model.query(query)
            print(answer['answer'])
            print("Excerpt searched:\n")
            print('Score:', answer['context']['score'])
            print('Text:\n', answer['context']['section'].text, '\n')
    else:
        for query in queries:
            print(f"\nQuery: \"{query}\"")
            print("Processing answer...")
            answer = rag_model.query(query)
            print(answer['answer'])
            print("Excerpt searched:\n")
            print('Score:', answer['context']['score'])
            print('Text:\n', answer['context']['section'].text, '\n')


if __name__ == '''__main__''':
    parser = argparse.ArgumentParser(description='An application that reads all PDFs in a provided directory, and answers questions about them.')
    parser.add_argument('--pdf-directory', type=str, required=True, help='Path to the PDF directory')
    parser.add_argument('--tesseract-path', type=str, required=True, help='Path to the Tesseract executable (e.g. /opt/homebrew/bin/tesseract)')
    parser.add_argument('--questions', metavar='Q', type=str, nargs='*', help='A list of questions')

    args = parser.parse_args()
    questions = args.questions if args.questions else []

    main(args.pdf_directory, args.tesseract_path, questions)