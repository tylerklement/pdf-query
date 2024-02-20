from llmware.prompts import Prompt


class RAGModel:
    '''
    Class which accepts a vector index and enables querying the index with questions.
    '''
    def __init__(self, vector_index, model_name="llmware/bling-1b-0.1"):
        self.prompter = Prompt().load_model(model_name)
        self.vector_index = vector_index

    def query(self, query, temperature=0.0):
        '''
        Main query method to ask a question regarding the index documents.
        '''
        retrieved_result = self.vector_index.search(query, n=1)[0]
        output = self.prompter.prompt_main(query,
                                           context=retrieved_result['section'].text,
                                           prompt_name="default_with_context",
                                           temperature=temperature)
        return {'answer': output['llm_response'].strip(), 'context': retrieved_result}
