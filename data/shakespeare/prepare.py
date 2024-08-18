import os
import requests
#import tiktoken
from transformers import AutoTokenizer
import numpy as np
import pickle
from transformers import GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

repo = load_dataset("YvanCarre/cnn-news-anonymised")

with open(input_file_path, "w") as corpus:
    for dataset_name in repo: 
        for text in tqdm(repo[dataset_name], desc=f"Processing {dataset_name}"): # Access dataset by name
            corpus.write(text["article"] + "\n")

print("completed")



with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe

#enc = tiktoken.get_encoding("gpt2")
#train_ids = enc.encode_ordinary(train_data)
#val_ids = enc.encode_ordinary(val_data)
tokenizer = AutoTokenizer.from_pretrained("YvanCarre/anonym-tokenizer")
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def process(example):

    ids = tokenizer.encode(example['article'], add_special_tokens=False)  
    ids.append(tokenizer.eos_token_id)  

    out = {'ids': ids, 'len': len(ids)}
    return out

tokenized = repo.map(
        process,
        remove_columns=['article','highlights', 'id'],
        desc="tokenizing the splits",
        num_proc=8,
    )

for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()






# train_ids = tokenizer.encode(train_data, add_special_tokens=False)
# val_ids = tokenizer.encode(val_data, add_special_tokens=False)


# print(f"train has {len(train_ids):,} tokens")
# print(f"val has {len(val_ids):,} tokens")

# # export to bin files
# train_ids = np.array(train_ids, dtype=np.uint16)
# val_ids = np.array(val_ids, dtype=np.uint16)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


vocab_size = 32001
meta = {
    'vocab_size': vocab_size,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)