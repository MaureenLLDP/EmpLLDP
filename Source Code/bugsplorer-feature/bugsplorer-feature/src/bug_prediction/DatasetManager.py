import math
import os
import pickle
import time
from datetime import timedelta
from pathlib import Path
from typing import Type, Protocol

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


class InfoCallback(Protocol):
    def __call__(self, msg: str, *args) -> None:
        ...


class DatasetManager:
    __slots__ = (
        "dataset_path",
        "target",
        "split_names",
        "cache_dir",
        "tokenizer",
        "info",
        "max_line_len",
        "max_file_len",
    )
    NUM_TOKENS_IN_4_LINES = 64
    DATASET_PAGE_SIZE = 10_000
    CACHE_FILE_EXT = ".pt"
    
    FEATURE_DIR = "dataset/linedp/handcrafted_features/"

    ALL_PROJECTS = {
        "activemq": ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
        "camel": ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
        "derby": ["derby-10.2.1.6", "derby-10.3.1.4", "derby-10.5.1.1"],
        "groovy": ["groovy-1_5_7", "groovy-1_6_BETA_1", "groovy-1_6_BETA_2"],
        "hbase": ["hbase-0.94.0", "hbase-0.95.0", "hbase-0.95.2"],
        "hive": ["hive-0.9.0", "hive-0.10.0", "hive-0.12.0"],
        "jruby": ["jruby-1.1", "jruby-1.4.0", "jruby-1.5.0", "jruby-1.7.0.preview1"],
        "lucene": ["lucene-2.3.0", "lucene-2.9.0", "lucene-3.0.0", "lucene-3.1"],
        "wicket": ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"]
    }
    
    HANDCRAFTED_FEATURES = [
        'function', 'recursive_function', 'blocks_count', 'recursive_blocks_count',
        'for_block', 'do_block', 'while_block', 'if_block', 'switch_block', 'conditional_count',
        'literal_string', 'integer_literal', 'literal_count', 'variable_count',
        'if_statement', 'for_statement', 'while_statement', 'do_statement', 'switch_statement',
        'conditional_and_loop_count', 'variable_declaration', 'function_declaration_count',
        'variable_declaration_statement', 'declaration_count', 'pointer_count',
        'user_defined_function_count', 'function_call_count', 'binary_operator',
        'unary_operator', 'compound_assignment_count', 'operator_count', 'array_usage'
    ]

    def __init__(
        self,
        tokenizer_class: Type[PreTrainedTokenizerFast],
        tokenizer_name: str,
        dataset_path: str,
        target: str,
        split_names: tuple[str, ...],
        cache_dir: str,
        info: InfoCallback,
        max_line_len,
        max_file_len,
    ):
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        self.dataset_path = dataset_path
        assert target in ["line", "file"]
        self.target = target
        self.split_names = split_names
        self.cache_dir = f"{cache_dir}-{max_file_len}-{max_line_len}"
        self.info = info
        self.max_line_len = max_line_len
        self.max_file_len = max_file_len

    def load_dataset(self) -> tuple[TensorDataset, ...]:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        assert os.path.isdir(self.dataset_path)
        cache_filepath_for, filepath_for = self._get_file_and_cache_path()

        last_modified_time_filepath = os.path.join(self.cache_dir, "last_modified_time.txt")

        if os.path.exists(last_modified_time_filepath):
             pass 

        self.info("Reading dataset and merging features...")
        tensor_datasets = self._read_and_cache(
            filepath_for,
            cache_filepath_for,
            last_modified_time_filepath,
            time.time(),
        )
        return tensor_datasets

    def _get_file_and_cache_path(self):
        filepath_for = {}
        cache_filepath_for = {}
        for filename in os.listdir(self.dataset_path):
            split_name = filename.split(".")[0]
            if split_name in self.split_names:
                filepath_for[split_name] = os.path.join(self.dataset_path, filename)
                cache_filepath_for[split_name] = os.path.join(
                    self.cache_dir, f"{split_name}{self.CACHE_FILE_EXT}"
                )
        return cache_filepath_for, filepath_for

    def _try_reading_cache(self, cache_filepath_for, last_modified_time_filepath, last_modified_time):
        if os.path.exists(last_modified_time_filepath):
            with open(last_modified_time_filepath, "r", encoding="utf8") as f:
                prev = float(f.read())
            if math.isclose(prev, last_modified_time, abs_tol=1e-6):
                if all(os.path.exists(p) for p in cache_filepath_for.values()):
                    self.info("Loading dataset from cache")
                    return tuple(torch.load(cache_filepath_for[split]) for split in self.split_names)
        return None

    def _read_and_cache(
        self,
        filepath_for,
        cache_filepath_for,
        last_modified_time_filepath,
        last_modified_time,
    ) -> tuple[TensorDataset, ...]:
        tick = time.perf_counter_ns()
        
        split_dataframes: dict[str, pd.DataFrame] = {
            split_name: pd.read_parquet(filepath_for[split_name])
            for split_name in self.split_names
        }

        split_dataframes = self._load_and_merge_handcrafted_features(split_dataframes)

        self._normalize_handcrafted_features(split_dataframes)

        if "linedp" in self.dataset_path:
            split_tensor_datasets = {
                split_name: self._tokenize_linedp(split_dataframe, split_name=split_name)
                for split_name, split_dataframe in split_dataframes.items()
            }
        else:
            split_tensor_datasets = {
                split_name: self._tokenize_defector(split_dataframe, split_name=split_name)
                for split_name, split_dataframe in split_dataframes.items()
            }

        for split_name, (tensor_dataset, *file_content) in split_tensor_datasets.items():
            torch.save(tensor_dataset, cache_filepath_for[split_name])
            file_content_file_path = cache_filepath_for[split_name].replace(".pt", ".pickle")
            with open(file_content_file_path, "wb") as f:
                pickle.dump(file_content, f)

        with open(last_modified_time_filepath, "w", encoding="utf8") as f:
            f.write(str(last_modified_time))

        tock = time.perf_counter_ns()
        elapsed = timedelta(microseconds=(tock - tick) / 1000)
        self.info(f"Tokenized dataset in {elapsed}")

        return tuple(tensor_dataset for tensor_dataset, *_ in split_tensor_datasets.values())

    def _load_and_merge_handcrafted_features(self, split_dataframes):

        self.info("Loading all handcrafted feature CSVs...")
        
        all_features_list = []
        
        for project, releases in self.ALL_PROJECTS.items():
            for release in releases:
                csv_path = os.path.join(self.FEATURE_DIR, f"{release}_ast_features.csv")
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if 'filename' in df.columns and 'line_number' in df.columns:
                            mapping = self._align_feature_columns(df.columns)
                            cols_to_keep = ['filename', 'line_number'] + list(mapping.keys())
                            df = df[cols_to_keep].rename(columns=mapping)
                            all_features_list.append(df)
                        else:
                            self.info(f"Warning: {csv_path} missing filename or line_number column.")
                    except Exception as e:
                        self.info(f"Error reading {csv_path}: {e}")
                else:
                    pass

        if not all_features_list:
            raise ValueError(f"No feature files found in {self.FEATURE_DIR}. Please check path.")

        self.info("Concatenating feature DataFrames...")
        full_features_df = pd.concat(all_features_list, ignore_index=True)
        
        full_features_df.drop_duplicates(subset=['filename', 'line_number'], keep='first', inplace=True)

        self.info(f"Loaded {len(full_features_df)} lines of features.")

        merged_splits = {}
        for split_name, split_df in split_dataframes.items():
            self.info(f"Merging features into {split_name} split...")
            

            merged_df = pd.merge(
                split_df, 
                full_features_df, 
                on=['filename', 'line_number'], 
                how='left'
            )
            

            missing_count = merged_df[self.HANDCRAFTED_FEATURES[0]].isna().sum()
            if missing_count > 0:
                self.info(f"Warning: {missing_count} lines in {split_name} did not match any features. Filling with 0.")

                merged_df[self.HANDCRAFTED_FEATURES] = merged_df[self.HANDCRAFTED_FEATURES].fillna(0)
            
            merged_splits[split_name] = merged_df

        return merged_splits

    def _align_feature_columns(self, actual_columns: pd.Index) -> dict[str, str]:

        mapping = {}
        used_actuals = set()
        
        for expected in self.HANDCRAFTED_FEATURES:
            match = None
            if expected in actual_columns:
                match = expected
            else:
                candidates = [col for col in actual_columns if expected.startswith(col) and col not in used_actuals]
                candidates = [c for c in candidates if c not in ['filename', 'line_number', 'file_label', 'line_label']]
                
                if candidates:
                    match = max(candidates, key=len)
            
            if match:
                mapping[match] = expected
                used_actuals.add(match)
        
        return mapping

    def _normalize_handcrafted_features(self, split_dataframes: dict[str, pd.DataFrame]):
        self.info("Normalizing handcrafted features...")
        train_df = split_dataframes.get('train', next(iter(split_dataframes.values())))

        train_data = train_df[self.HANDCRAFTED_FEATURES].fillna(0).values.astype(np.float32)
        
        min_vals = np.min(train_data, axis=0)
        max_vals = np.max(train_data, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-6 

        for split_name, df in split_dataframes.items():
            data = df[self.HANDCRAFTED_FEATURES].fillna(0).values.astype(np.float32)
            normalized_data = (data - min_vals) / range_vals
            df['handcrafted_vec'] = list(normalized_data)

    def _tokenize_linedp(self, dataframe: pd.DataFrame, split_name: str):
        return self._tokenize_linedp_impl(dataframe, split_name)

    def _tokenize_linedp_impl(self, dataframe: pd.DataFrame, split_name: str):
        if "repo" not in dataframe.columns:
            dataframe["repo"] = dataframe["filename"].apply(
                lambda x: x.split('-')[0].split('_')[0] if isinstance(x, str) else "unknown"
            )
        dataframe["code_line"] = dataframe["code_line"].fillna("").astype(str)
        dataframe = dataframe.reset_index(drop=True)
        dataframe['global_line_idx'] = dataframe.index
        dataframe['file_id'] = (dataframe["repo"] + "_" + dataframe["filename"]).astype('category').cat.codes

        dataframe = dataframe.groupby(dataframe["repo"] + dataframe["filename"], sort=False)
        dataframe.apply(lambda grp: grp.sort_values("line_number", inplace=True))

        file_content = dataframe["code_line"].apply(
            lambda grp: grp.str.replace(self.tokenizer.eos_token, "", regex=False).to_list()
        ).to_list()
        file_info = dataframe[["repo", "filename"]].first().values.tolist()

        tokenization_progress = tqdm(
            [(lines, self.tokenizer, self.max_line_len, self.max_file_len) for lines in file_content],
            desc=f"Tokenizing {split_name}",
            smoothing=0.001,
        )
        file_tensors = tuple(
            file_chunk for file_chunks in map(_tokenize_lines, tokenization_progress) for file_chunk in file_chunks
        )
        source_tensor = torch.stack(file_tensors)

        buggy_lines_of_file = dataframe["line-label"].apply(list).to_list()
        global_indices_of_file = dataframe["global_line_idx"].apply(list).to_list()
        file_id_of_file = dataframe["file_id"].apply(list).to_list()
        handcrafted_vectors_of_file = dataframe["handcrafted_vec"].apply(list).to_list()

        buggy_line_matrix, global_index_matrix, file_id_matrix, handcrafted_matrix = [], [], [], []

        zero_feature = [0.0] * 32 

        for buggy_lines, global_indices, file_ids, handcrafted_vecs in zip(
            buggy_lines_of_file, global_indices_of_file, file_id_of_file, handcrafted_vectors_of_file
        ):
            padding_len = self.max_file_len - len(buggy_lines)
            
            temp_buggy_chunks = []
            temp_idx_chunks = []
            temp_fid_chunks = []
            temp_feat_chunks = []

            if padding_len < 0: 
                for start in range(0, len(buggy_lines), self.max_file_len - 64):
                    end = start + self.max_file_len
                    
                    chunk_buggy = buggy_lines[start:end]
                    chunk_idx = global_indices[start:end]
                    chunk_fid = file_ids[start:end]
                    chunk_feat = handcrafted_vecs[start:end]

                    if len(chunk_buggy) < self.max_file_len:
                        pad = self.max_file_len - len(chunk_buggy)
                        chunk_buggy.extend([False] * pad)
                        chunk_idx.extend([-1] * pad)
                        chunk_fid.extend([-1] * pad)
                        chunk_feat.extend([zero_feature] * pad)
                    
                    temp_buggy_chunks.append(chunk_buggy)
                    temp_idx_chunks.append(chunk_idx)
                    temp_fid_chunks.append(chunk_fid)
                    temp_feat_chunks.append(chunk_feat)


                last_chunk_size = len(buggy_lines) % (self.max_file_len - 64)
                if len(temp_buggy_chunks) > 1 and last_chunk_size > 0 and last_chunk_size <= 64:
                    temp_buggy_chunks.pop()
                    temp_idx_chunks.pop()
                    temp_fid_chunks.pop()
                    temp_feat_chunks.pop()

                buggy_line_matrix.extend(temp_buggy_chunks)
                global_index_matrix.extend(temp_idx_chunks)
                file_id_matrix.extend(temp_fid_chunks)
                handcrafted_matrix.extend(temp_feat_chunks)

            else: 
                padding = [False] * padding_len
                padding_idx = [-1] * padding_len
                padding_feat = [zero_feature] * padding_len
                
                buggy_line_matrix.append(buggy_lines + padding)
                global_index_matrix.append(global_indices + padding_idx)
                file_id_matrix.append(file_ids + padding_idx)
                handcrafted_matrix.append(handcrafted_vecs + padding_feat)

        target_tensor = torch.tensor(np.array(buggy_line_matrix), dtype=torch.bool)
        global_index_tensor = torch.tensor(np.array(global_index_matrix), dtype=torch.long)
        file_id_tensor = torch.tensor(np.array(file_id_matrix), dtype=torch.long)
        handcrafted_tensor = torch.tensor(np.array(handcrafted_matrix), dtype=torch.float32)

        return TensorDataset(source_tensor, target_tensor, global_index_tensor, file_id_tensor, handcrafted_tensor), file_content, file_info

    def _truncate_or_pad(self, file_content):
        return [tuple(lines[:self.max_file_len]) if len(lines) > self.max_file_len else (*lines, *[""] * (self.max_file_len - len(lines))) for lines in file_content]

def _tokenize_lines(args) -> list[torch.Tensor]:
    lines, tokenizer, max_line_len, max_file_len = args
    line_token_ids = tokenizer(lines, max_length=max_line_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    split_line_token_ids = []
    for i in range(0, len(line_token_ids), max_file_len - 64):
        split_line_token_ids.append(line_token_ids[i : i + max_file_len])
    if len(split_line_token_ids) > 1 and len(split_line_token_ids[-1]) <= 64:
        split_line_token_ids.pop()
    if len(split_line_token_ids[-1]) < max_file_len:
        num_padded = max_file_len - len(split_line_token_ids[-1])
        padding = torch.tensor([[tokenizer.pad_token_id]]).repeat(num_padded, max_line_len)
        split_line_token_ids[-1] = torch.cat([split_line_token_ids[-1], padding], dim=0)
    return split_line_token_ids