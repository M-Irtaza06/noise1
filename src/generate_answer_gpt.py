import os 
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *
from prompt_dataset import PromptDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "data_path": 'data/10k_train_dataset.json',
    "random_results_path": "data/10k_random_results_at60.pkl",
    "adore_search_results_path": "data/adore_search_results_at200.pkl",
    "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run GPT Generation.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='gpt2', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=1024)
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)    
    parser.add_argument('--use_random', type=str2bool, help='Use random irrelevant documents')
    parser.add_argument('--use_adore', type=str2bool, help="Use the retrieved documents from ADORE", default=False)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context')
    parser.add_argument('--get_documents_without_answer', type=str2bool, help='Select only documents without the answer (e.g., distracting)', default=True)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=50)
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
    search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: AutoTokenizer
) -> DataLoader:
    
    prompt_ds = PromptDataset(
        corpus=corpus, data_path=info['data_path'], 
        tokenizer=tokenizer, 
        max_tokenized_length=args.model_max_length - 2, 
        search_results=search_results,
        full_to_subset_idx_map=full_to_subset_idx_map,
        do_normalize_query=True, 
        num_documents_in_context=args.num_documents_in_context,
        gold_position=args.gold_position,
        get_documents_without_answer=args.get_documents_without_answer,
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
    print(f"MODEL: {args.llm_id}")
    print(f"USE RANDOM IN CONTEXT: {args.use_random}")
    print(f"USE ADORE: {args.use_adore}")
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_documents_in_context}")
    print(f"DOCUMENTS WITHOUT ANSWER: {args.get_documents_without_answer}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def generate_and_save(
    args: argparse.Namespace, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    prompt_dataloader: DataLoader
):
    num_doc = args.num_documents_in_context
    save_every = args.save_every
    gold_pos = args.gold_position
    retriever_str = "adore" if args.use_adore else "contriever"
    rand_str = "_rand" if args.use_random else ""
    answerless_str = "_answerless" if args.get_documents_without_answer else ""

    llm_folder = args.llm_id.split("/")[1] if '/' in args.llm_id else args.llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/train/classic/{retriever_str}/{num_doc}_doc"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    answer_string_in_prompt = "Answer:"

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.model_max_length).to(device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        generated_answers = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            all_info = []


def main():
    args = parse_arguments()

    print("Loading GPT model...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    
     # Set the padding side to 'left'
    tokenizer.padding_side = 'left'
    # Option 1: Set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Option 2: Add a custom padding token (commented out, you can use if needed)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.llm_id)

    # If you added a custom padding token, resize model embeddings
    # model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    print("GPT model loaded")

    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    search_results = load_search_results(args)
    print("Corpus and search results loaded")

    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, search_results, tokenizer
    )
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, model, tokenizer, prompt_dataloader)


if __name__ == "__main__":
    seed_everything(SEED)
    main()
