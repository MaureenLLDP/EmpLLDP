import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle

class BiLSTMDataset(Dataset):
    def __init__(self, word2vec, corpus_path, corpus_files, code_len, Train=True, use_manual_features=False, 
                 scaler_save_path=None):
        self.word2vec = word2vec
        self.code_len = code_len
        self.Train = Train
        self.use_manual_features = use_manual_features
        self.embedding_dim = self.word2vec.vector_size
        self.oov_vector = np.zeros(self.embedding_dim, dtype=np.float32)
        self.scaler_save_path = scaler_save_path or "./scaler.pkl"
        
        self.manual_feature_columns = [
            'function', 'recursive_function', 'blocks_count', 'recursive_blocks_count',
            'for_block', 'do_block', 'while_block', 'if_block', 'switch_block',
            'conditional_count', 'literal_string', 'integer_literal', 'literal_count',
            'variable_count', 'if_statement', 'for_statement', 'while_statement',
            'do_statement', 'switch_statement', 'conditional_and_loop_count',
            'variable_declaration', 'function_declaration_count', 'variable_declaration_statement',
            'declaration_count', 'pointer_count', 'user_defined_function_count',
            'function_call_count', 'binary_operator', 'unary_operator',
            'compound_assignment_count', 'operator_count', 'array_usage'
        ]

        self.scaler = None

        self.data_df = self._load_and_parse_ast_strings(corpus_path, corpus_files, Train)

        if self.use_manual_features:
            self._load_manual_features(corpus_path, corpus_files)

        if Train:
            print(f" {self.data_df['line_label'].sum()}")
            print(f" {len(self.data_df) - self.data_df['line_label'].sum()}")
            print(f" {self.data_df['line_label'].mean():.4f}")

    def _load_and_parse_ast_strings(self, corpus_path, corpus_files, Train):
        all_dfs = []
        for tmp_file in corpus_files:
            df_path = os.path.join(corpus_path, f"{tmp_file}.csv")
            if os.path.exists(df_path):
                df = pd.read_csv(df_path)
                empty_ast_count = sum(1 for ast in df['ast'] if ast == '[]')
                df = df[df['ast'] != '[]'].copy()
                
                df['filename'] = df['filename'].astype(str)
                df['corpus_file'] = tmp_file 
                all_dfs.append(df)
            else:
                print(f"Warning: File not found {df_path}")
                
        if not all_dfs:
            print("Error: No dataframes loaded for dataset.")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['line_label'] = combined_df['line_label'].astype(bool)
        combined_df['file_label'] = combined_df['file_label'].astype(bool)

        if Train:
            print(f"{combined_df['line_label'].sum()}")
            print(f"{len(combined_df) - combined_df['line_label'].sum()}")
        
        def parse_ast_string_to_list(ast_str):
            if ast_str.startswith('[') and ast_str.endswith(']'):
                nodes = [node.strip().replace("'", "").replace('"', '') for node in ast_str[1:-1].split(',')]
            else:
                nodes = [ast_str.strip().replace("'", "").replace('"', '')] if ast_str.strip() else []
            nodes = [node for node in nodes if node]
            return nodes

        combined_df['ast_nodes_list'] = combined_df['ast'].apply(parse_ast_string_to_list)
        
        return combined_df

    def _load_manual_features(self, corpus_path, corpus_files):
        manual_path = corpus_path.replace('preprocessed_data3', 'ast_feature_original')
        
        all_manual_features = []
        
        for tmp_file in corpus_files:
            manual_csv = os.path.join(manual_path, f"{tmp_file}_ast_features.csv")
            if os.path.exists(manual_csv):
                manual_df = pd.read_csv(manual_csv)
                manual_df['corpus_file'] = tmp_file 
                all_manual_features.append(manual_df)
            else:
                print(f"Warning: Manual features not found {manual_csv}")
        
        if all_manual_features:
            combined_manual = pd.concat(all_manual_features, ignore_index=True)
            
            missing_cols = [col for col in self.manual_feature_columns if col not in combined_manual.columns]
            if missing_cols:
                print(f"Warning: Missing manual feature columns: {missing_cols}")
                for col in missing_cols:
                    combined_manual[col] = 0
            
           
            self.data_df = self.data_df.merge(
                combined_manual[['filename', 'line_number', 'corpus_file'] + self.manual_feature_columns],
                on=['filename', 'line_number', 'corpus_file'],
                how='left'
            )
            
          
            for col in self.manual_feature_columns:
                non_null_count = self.data_df[col].notna().sum()
                null_count = self.data_df[col].isna().sum()
                if null_count > 0:
                    print(f" {col}:non_null {non_null_count}, null {null_count}")
            

            original_na_count = self.data_df[self.manual_feature_columns].isna().sum().sum()
            for col in self.manual_feature_columns:
                self.data_df[col] = self.data_df[col].fillna(0)

            self._standardize_manual_features()
            
            print("Manual features loaded, merged, and MinMax standardized.")

            feature_data = self.data_df[self.manual_feature_columns].values.astype(np.float32)

            non_zero_features = (feature_data != 0).any(axis=0)
            non_zero_count = non_zero_features.sum()

        else:
            for col in self.manual_feature_columns:
                self.data_df[col] = 0
            print("No manual features found, defaulting to 0.")

    def _standardize_manual_features(self):
        feature_data = self.data_df[self.manual_feature_columns].values.astype(np.float32)
        
        if self.Train:
            self.scaler = MinMaxScaler()
            feature_data_scaled = self.scaler.fit_transform(feature_data)

            with open(self.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
        else:
            if os.path.exists(self.scaler_save_path):
                with open(self.scaler_save_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                feature_data_scaled = self.scaler.transform(feature_data)
            else:
                feature_data_scaled = feature_data
        
        for i, col in enumerate(self.manual_feature_columns):
            self.data_df[col] = feature_data_scaled[:, i]


    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        row = self.data_df.iloc[item]
        
        ast_nodes = row['ast_nodes_list']

        embedded_nodes = []
        for node in ast_nodes:
            if node in self.word2vec.wv:
                embedded_nodes.append(self.word2vec.wv[node])
            else:
                embedded_nodes.append(self.oov_vector)
        
        if len(embedded_nodes) > self.code_len:
            embedded_nodes = embedded_nodes[:self.code_len]
        
        if len(embedded_nodes) < self.code_len:
            padding_needed = self.code_len - len(embedded_nodes)
            padding_vectors = [self.oov_vector for _ in range(padding_needed)]
            embedded_nodes.extend(padding_vectors)
        
        ast_embeddings = np.array(embedded_nodes, dtype=np.float32)
        
        if self.use_manual_features:
            manual_features = []
            for col in self.manual_feature_columns:
                manual_features.append(float(row[col]))
            manual_features = np.array(manual_features, dtype=np.float32)  # [32]
            
            manual_features_broadcasted = np.tile(manual_features, (self.code_len, 1))
            
            ast_input = np.concatenate([ast_embeddings, manual_features_broadcasted], axis=1)
        else:
            ast_input = ast_embeddings
        
        ast_input = torch.tensor(ast_input, dtype=torch.float32)
        is_defect = torch.tensor(int(row['line_label']), dtype=torch.float32)

        output = {
            "ast_input": ast_input,  # [code_len, 532] if use_fusion else [code_len, 500]
            "is_defect": is_defect,
            "filename": row['filename'],
            "line_number": row['line_number'],
            "file_label": row['file_label']
        }
        
        return output