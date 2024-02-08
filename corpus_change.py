import argparse  
import os
from beir import util
import openai
import json
import concurrent.futures


from beir.datasets.data_loader import GenericDataLoader



class CorpusChange:
    def __init__(self, dataset, subfolder, prompt1, prompt2, path):
        self.dataset = dataset
        self.subfolder = subfolder
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.path = path

    def load_data(self):
        ''' 
        subfolder can be train or test
        '''
        print("Dataset is:", self.dataset)
        
        # Download and unzip dataset
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(self.dataset)
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        print("Dataset downloaded here: {}".format(data_path))

        data_path = "datasets/" + self.dataset
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=self.subfolder)

        return corpus, queries, qrels

    def llm_inference1(self, corpus, path):
        # Perform language model inference for prompt1
        texts_to_ids = {}   
        prompt_format = "'{}' \n {}"

        for id_, info in corpus.items():
            text = info["text"]
            prompt = prompt_format.format(text, self.prompt1)
            texts_to_ids[text] = id_

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.send_to_gpt, prompt_format.format(text, self.prompt1), id_, path, "llm1") for text, id_ in texts_to_ids.items()]

        concurrent.futures.wait(futures)
    
    def llm_inference2(self, corpus, path):
        # Perform language model inference for prompt2
        texts_to_ids = {}   
        prompt_format = "'{}' \n {}"

        for id_, info in corpus.items():
            text = info["text"]
            prompt = prompt_format.format(text, self.prompt2)
            texts_to_ids[text] = id_

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.send_to_gpt, prompt_format.format(text, self.prompt2), id_, path, "llm2") for text, id_ in texts_to_ids.items()]

        concurrent.futures.wait(futures)

    def send_to_gpt(self, prompt, id_, saved_path, control):
        # Send prompts to GPT-3.5 for completion
        print("entered here")
        
        chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}])
        response = chat_completion.choices[0].message.content
        print(response)

        if control == "llm1":
            # Save results for prompt1
            print("entered if condition")
            self.results1[id_] = response
            with open(saved_path, "w") as json_file:
                json.dump(self.results1, json_file)
        else:
            # Save results for prompt2
            print("entered else condition")
            self.results2[id_] = response
            with open(saved_path, "w") as json_file:
                json.dump(self.results2, json_file)
        
    def change_corpus(self):
        # Main function to change corpus
        self.results1 = {}
        self.results2 = {}
        saved_path = "corpus_v1_.json"
        saved_path2 = "keywords_.json"
        
        corpus, queries, qrels = self.load_data()

        '''short_corpus = {}
        i = 0
        for key, info in corpus.items():
            if i < 3:
                short_corpus[key] = info
            i += 1 '''

        self.llm_inference1(corpus, saved_path)

        print("self-results1:", self.results1)
        new_corpus = corpus.copy()
        for key, info in new_corpus.items():
            new_corpus[key]['text'] = self.results1[key]

        with open("/cta/users/iaktemur/rag/scifact/experiment/new_corpus_v1.json", "w") as json_file:
            json.dump(new_corpus, json_file)

        self.llm_inference2(new_corpus, saved_path2)
        for key, info in new_corpus.items():
            new_corpus[key]['text'] = new_corpus[key]['text'] + "\n Keywords and synonyms: " + self.results2[key]

        with open(self.path, "w") as json_file:
            json.dump(new_corpus, json_file)
 
def main():
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', help='Dataset name', required=True)
    parser.add_argument('--subfolder', help='Subfolder name', required=True)
    parser.add_argument('--prompt1', help='prompt to get facts?', required=True)
    parser.add_argument('--prompt2', help='prompt to get keywords?', required=True)
    parser.add_argument('--path', help='path to save new corpus', default="/cta/users/iaktemur/rag/scifact/experiment/new_corpus_v2.json")
    args = parser.parse_args()

    corpus_change = CorpusChange(args.dataset, args.subfolder, args.prompt1, args.prompt2, args.path)
    corpus_change.change_corpus()

if __name__ == "__main__":
    main()
