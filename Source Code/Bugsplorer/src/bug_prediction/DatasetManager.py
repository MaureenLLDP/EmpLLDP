import math
import os
import pickle
import time
from datetime import timedelta
from pathlib import Path
from typing import Type, Protocol

import pandas as pd
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
        """
        This function loads the dataset from the cache if it exists.
        Otherwise, it reads the dataset from the file and caches it.
        """
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        assert os.path.isdir(self.dataset_path)
        cache_filepath_for, filepath_for = self._get_file_and_cache_path()

        last_modified_time_filepath = os.path.join(
            self.cache_dir, "last_modified_time.txt"
        )
        last_modified_time = max(map(os.path.getmtime, filepath_for.values()))
        cached_dataset = self._try_reading_cache(
            cache_filepath_for,
            last_modified_time_filepath,
            last_modified_time,
        )
        if cached_dataset:
            return cached_dataset

        self.info("Reading dataset.")

        tensor_datasets = self._read_and_cache(
            filepath_for,
            cache_filepath_for,
            last_modified_time_filepath,
            last_modified_time,
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

    def _try_reading_cache(
        self,
        cache_filepath_for: dict[str, str],
        last_modified_time_filepath: str,
        last_modified_time: float,
    ):
        if os.path.exists(last_modified_time_filepath):
            with open(
                last_modified_time_filepath, "r", encoding="utf8"
            ) as last_modified_time_file:
                prev_last_modified_time = float(last_modified_time_file.read())
            if math.isclose(prev_last_modified_time, last_modified_time, abs_tol=1e-6):
                self.info("Loading dataset from cache")
                dataset_splits = tuple(
                    torch.load(cache_filepath_for[split_name])
                    for split_name in self.split_names
                )

                return dataset_splits

            self.info("Cache outdated.")

        self.info("Cache miss.")

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

        split_tensor_datasets: dict[str, tuple[TensorDataset, list[list[str]]]] = {
            split_name: self._tokenize_linedp(split_dataframe, split_name=split_name)
            for split_name, split_dataframe in split_dataframes.items()
        }

        for split_name, (
            tensor_dataset,
            *file_content,
        ) in split_tensor_datasets.items():
            torch.save(tensor_dataset, cache_filepath_for[split_name])

            file_content_file_path = cache_filepath_for[split_name].replace(
                ".pt", ".pickle"
            )
            with open(file_content_file_path, "wb") as file_content_file:
                pickle.dump(file_content, file_content_file)

        with open(
            last_modified_time_filepath, "w", encoding="utf8"
        ) as last_modified_time_file:
            last_modified_time_file.write(str(last_modified_time))

        tock = time.perf_counter_ns()
        elapsed = timedelta(microseconds=(tock - tick) / 1000)
        self.info(f"Tokenized dataset in {elapsed}")

        return tuple(
            tensor_dataset for tensor_dataset, *_ in split_tensor_datasets.values()
        )

    def _tokenize_linedp(self, dataframe: pd.DataFrame, split_name: str):
        """
        修改版：添加原始索引和文件ID跟踪
        """
        # 确保存在 repo 列（兼容旧数据）
        if "repo" not in dataframe.columns:
            dataframe["repo"] = dataframe["filename"].apply(
                lambda x: x.split('-')[0].split('_')[0] if isinstance(x, str) else "unknown"
            )

        # 确保 code_line 是有效字符串
        dataframe["code_line"] = dataframe["code_line"].fillna("").astype(str)

        # 添加全局行索引（用于跟踪原始位置）
        dataframe = dataframe.reset_index(drop=True)
        dataframe['global_line_idx'] = dataframe.index
        
        # 添加文件ID（用于文件级IFA计算）
        dataframe['file_id'] = (dataframe["repo"] + "_" + dataframe["filename"]).astype('category').cat.codes

        dataframe = dataframe.groupby(
            dataframe["repo"] + dataframe["filename"], sort=False
        )
        dataframe.apply(lambda grp: grp.sort_values("line_number", inplace=True))

        # file_content contains list[list[line_of_code]]
        file_content = (
            dataframe["code_line"]
            .apply(
                lambda grp: grp.str.replace(
                    self.tokenizer.eos_token, "", regex=False
                ).to_list()
            )
            .to_list()
        )
        file_info = dataframe[["repo", "filename"]].first().values.tolist()

        tokenization_progress = tqdm(
            [
                (lines, self.tokenizer, self.max_line_len, self.max_file_len)
                for lines in file_content
            ],
            desc=f"Tokenizing {split_name}",
            smoothing=0.001,
        )
        file_tensors = tuple(
            file_chunk
            for file_chunks in map(_tokenize_lines, tokenization_progress)
            for file_chunk in file_chunks
        )
        source_tensor = torch.stack(file_tensors)
        self.info(f"Tokenization complete: {source_tensor.shape=}")
        assert source_tensor.shape[1:] == (
            self.max_file_len,
            self.max_line_len,
        )

        buggy_lines_of_file = (
            dataframe["line-label"]
            .apply(lambda grp: grp.to_list())
            .to_list()
        )
        
        # **关键修改：同时获取全局行索引和文件ID**
        global_indices_of_file = (
            dataframe["global_line_idx"]
            .apply(lambda grp: grp.to_list())
            .to_list()
        )
        
        file_ids_of_file = (
            dataframe["file_id"]
            .apply(lambda grp: grp.to_list())
            .to_list()
        )

        if self.target == "line":
            buggy_line_matrix = []
            global_index_matrix = []  # **新增：跟踪每个样本对应的原始全局索引**
            file_id_matrix = []  # **新增：跟踪每个样本对应的文件ID**
            
            for buggy_lines, global_indices, file_ids in zip(buggy_lines_of_file, global_indices_of_file, file_ids_of_file):
                padding_len = self.max_file_len - len(buggy_lines)
                if padding_len < 0:
                    # 分割成多个chunk
                    new_buggy_lines = []
                    new_global_indices = []  # **新增**
                    new_file_ids = []  # **新增**
                    
                    for start in range(0, len(buggy_lines), self.max_file_len - 64):
                        new_buggy_lines.append(
                            buggy_lines[start : start + self.max_file_len]
                        )
                        new_global_indices.append(
                            global_indices[start : start + self.max_file_len]
                        )
                        new_file_ids.append(
                            file_ids[start : start + self.max_file_len]
                        )

                    if len(new_buggy_lines) > 1 and len(new_buggy_lines[-1]) <= 64:
                        new_buggy_lines.pop()
                        new_global_indices.pop()
                        new_file_ids.pop()
                        
                    if len(new_buggy_lines[-1]) < self.max_file_len:
                        padding_size = self.max_file_len - len(new_buggy_lines[-1])
                        padding = [False] * padding_size
                        padding_idx = [-1] * padding_size  # 填充位置用-1标记
                        new_buggy_lines[-1].extend(padding)
                        new_global_indices[-1].extend(padding_idx)
                        new_file_ids[-1].extend(padding_idx)
                        
                    buggy_line_matrix.extend(new_buggy_lines)
                    global_index_matrix.extend(new_global_indices)
                    file_id_matrix.extend(new_file_ids)

                else:
                    padding = [False] * padding_len
                    padding_idx = [-1] * padding_len
                    buggy_line_matrix.append([*buggy_lines, *padding])
                    global_index_matrix.append([*global_indices, *padding_idx])
                    file_id_matrix.append([*file_ids, *padding_idx])

            target_tensor = torch.tensor(buggy_line_matrix)
            global_index_tensor = torch.tensor(global_index_matrix)  # **新增**
            file_id_tensor = torch.tensor(file_id_matrix)  # **新增**

            assert target_tensor.shape[1:] == (self.max_file_len,), target_tensor.shape
            assert (
                source_tensor.shape[0] == target_tensor.shape[0]
            ), f"{source_tensor.shape=} {target_tensor.shape=}"

            # **关键修改：将全局索引和文件ID也加入 TensorDataset**
            tensor_dataset = TensorDataset(source_tensor, target_tensor, global_index_tensor, file_id_tensor)
            
        elif self.target == "file":
            is_buggy_file = dataframe["line-label"].apply(lambda grp: grp.any())
            target_tensor = torch.tensor(is_buggy_file.to_list())
            assert target_tensor.shape == (len(dataframe),), target_tensor.shape
            tensor_dataset = TensorDataset(source_tensor, target_tensor)
        else:
            raise ValueError(f"Unknown target: {self.target}")

        fixed_len_file_content = self._truncate_or_pad(file_content)
        return tensor_dataset, fixed_len_file_content, file_info

    def _truncate_or_pad(self, file_content):
        return [
            tuple(lines[: self.max_file_len])
            if len(lines) > self.max_file_len
            else (*lines, *[""] * (self.max_file_len - len(lines)))
            for lines in file_content
        ]


def _tokenize_lines(args) -> list[torch.Tensor]:
    lines, tokenizer, max_line_len, max_file_len = args
    line_token_ids = tokenizer(
        lines,
        max_length=max_line_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    
    split_line_token_ids = []
    for i in range(0, len(line_token_ids), max_file_len - 64):
        split_line_token_ids.append(line_token_ids[i : i + max_file_len])
    if len(split_line_token_ids) > 1 and len(split_line_token_ids[-1]) <= 64:
        split_line_token_ids.pop()

    if len(split_line_token_ids[-1]) < max_file_len:
        num_padded_lines = max_file_len - len(split_line_token_ids[-1])
        line_padding = torch.tensor([[tokenizer.pad_token_id]]).repeat(
            num_padded_lines, max_line_len
        )
        split_line_token_ids[-1] = torch.cat(
            [
                split_line_token_ids[-1],
                line_padding,
            ],
            dim=0,
        )

    return split_line_token_ids