import os
import argparse
import warnings
from tqdm import tqdm
import openai
import pandas as pd
from typing import Tuple, Dict, Optional, List
from torch.utils.data import DataLoader

from utils import *
from prompt_dataset import PromptDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
SEED = 10

info = {
    "data_path": 'data/10k_train_dataset.json',
    "random_results_path": "data/10k_random_results_at60.pkl",
    "adore_search_results_path": "data/adore_search_results_at200.pkl",
    "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
}

class DummyTokenizer:
    def __init__(self, name_or_path="dummy-tokenizer"):
        self.name_or_path = name_or_path
    
    def encode(self, text, **kwargs):
        # Simple tokenization by splitting text into words
        return text.split()

    def decode(self, tokens, **kwargs):
        # Recombine tokens into text
        return ' '.join(tokens)

    def __call__(self, text, **kwargs):
        return self.encode(text, **kwargs)

    def tokenize(self, text):
        # Mimic tokenization by splitting text into words
        return text.split()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run GPT Generation via OpenAI API.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='OpenAI GPT model identifier')
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)    
    parser.add_argument('--use_random', type=str2bool, help='Use random irrelevant documents')
    parser.add_argument('--use_adore', type=str2bool, help="Use the retrieved documents from ADORE", default=False)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context')
    parser.add_argument('--get_documents_without_answer', type=str2bool, help='Select only documents without the answer (e.g., distracting)', default=True)
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens to generate', default=50)
    parser.add_argument('--batch_size', type=int, help='Batch size for generation')
    parser.add_argument('--save_every', type=int, default=250)

    args = parser.parse_args()

    if args.num_documents_in_context is None:
        parser.error("'num_documents_in_context' must be specified.")
    if args.num_documents_in_context <= 0:
        parser.error("'num_documents_in_context' must be a positive integer.")
    if args.gold_position is not None and (args.gold_position < 0 or args.gold_position >= args.num_documents_in_context):
        parser.error("'gold_position' must be within the range of 'num_documents_in_context'.")

    return args

def load_corpus(
    args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    # Load the corpus
    if args.load_full_corpus:
        corpus = read_corpus_json('data/corpus.json')
        return corpus, None

    if args.use_random:
        corpus, full_to_subset_idx_map = read_corpus_with_random()
    elif args.use_adore:
        corpus, full_to_subset_idx_map = read_corpus_with_adore()
    else: 
        # Corpus with documents from Contriever
        corpus, full_to_subset_idx_map = read_corpus_with_contriever()

    return corpus, full_to_subset_idx_map

def load_search_results(args: argparse.Namespace) -> List[Tuple[List[int], List[float]]]:
    # Decide on search results path based on conditions
    if args.use_random:
        search_results_path = info['random_results_path']
    elif args.use_adore:
        search_results_path = info['adore_search_results_path']
    else:
        # Search results from Contriever
        search_results_path = info['contriever_search_results_path'] 

    search_results = read_pickle(search_results_path)
    return search_results

def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    search_results: List[Tuple[List[int], List[float]]]
) -> DataLoader:
    
    tokenizer = DummyTokenizer()  # Using DummyTokenizer to satisfy the tokenizer requirement
    
    prompt_ds = PromptDataset(
        corpus=corpus, data_path=info['data_path'], 
        max_tokenized_length=2048 - 2,  # Adjust depending on your prompt structure
        search_results=search_results,
        full_to_subset_idx_map=full_to_subset_idx_map,
        do_normalize_query=True, 
        num_documents_in_context=args.num_documents_in_context,
        gold_position=args.gold_position,
        get_documents_without_answer=args.get_documents_without_answer,
        tokenizer=tokenizer  # Pass the dummy tokenizer
    )
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return prompt_dataloader

def print_info(args: argparse.Namespace):
    print("INFO:")
    print(f"DATA: {info['data_path']}")
    print(f"MODEL: {args.model}")
    print(f"USE RANDOM IN CONTEXT: {args.use_random}")
    print(f"USE ADORE: {args.use_adore}")
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_documents_in_context}")
    print(f"DOCUMENTS WITHOUT ANSWER: {args.get_documents_without_answer}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")

def generate_text(prompt: str, model: str, max_tokens: int) -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message['content'].strip()

def generate_and_save(
    args: argparse.Namespace, 
    prompt_dataloader: DataLoader
):
    num_doc = args.num_documents_in_context
    save_every = args.save_every
    gold_pos = args.gold_position
    retriever_str = "adore" if args.use_adore else "contriever"
    rand_str = "_rand" if args.use_random else ""
    answerless_str = "_answerless" if args.get_documents_without_answer else ""

    llm_folder = args.model
    saving_dir = f"{args.output_dir}/{llm_folder}/train/classic/{retriever_str}/{num_doc}_doc"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        generated_answers = [generate_text(prompt, args.model, args.max_tokens) for prompt in prompts]
        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            all_info = []

def main():
    args = parse_arguments()

    openai.api_key = args.api_key

    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    search_results = load_search_results(args)
    print("Corpus and search results loaded")

    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, search_results
    )
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, prompt_dataloader)

if __name__ == "__main__":
    seed_everything(SEED)
    main()
