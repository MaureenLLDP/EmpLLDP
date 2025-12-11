import torch
import os, argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from BAFLineDP import *
from my_util import *

class InputFeatures(object):
    def __init__(self, input_ids, label, static_features=None):
        self.input_ids = input_ids
        self.label = label
        self.static_features = static_features

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, datasets, labels, static_features_list=None):
        self.examples = []
        labels = torch.FloatTensor(labels)
        
        if static_features_list is None:
            static_features_list = [None] * len(datasets)
            
        for dataset, label, static_features in zip(datasets, labels, static_features_list):
            dataset_ids = [convert_examples_to_features(item, tokenizer, args) for item in dataset]
            self.examples.append(InputFeatures(dataset_ids, label, static_features))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.examples[i].static_features is not None:
            return (torch.tensor(self.examples[i].input_ids), 
                    self.examples[i].label, 
                    torch.tensor(self.examples[i].static_features, dtype=torch.float32))
        else:
            return torch.tensor(self.examples[i].input_ids), self.examples[i].label

def convert_examples_to_features(item, tokenizer, args):
    code = ' '.join(item)
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return source_ids

def load_and_process_static_features_for_prediction(static_features_df, dataset_name):
    """为预测阶段加载并预处理静态特征，使用简单的标准化"""
    if static_features_df is None:
        return None
    
    # 获取特征列（跳过前4列：filename, line_number, file_label, line_label）
    feature_cols = static_features_df.columns[4:].tolist()
    
    # 提取所有特征数据进行标准化
    feature_data = static_features_df[feature_cols].values
    
    # 使用MinMaxScaler进行标准化
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(feature_data)
    
    # 更新DataFrame中的特征数据
    static_features_df = static_features_df.copy()
    static_features_df[feature_cols] = normalized_features
    
    print(f"Processed static features for {dataset_name}:")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Feature range after normalization: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")
    
    return static_features_df

def preprocess_static_features(static_features_df):
    """预处理静态特征，建立快速查找索引"""
    if static_features_df is None:
        return None, 0
    
    feature_cols = static_features_df.columns[4:].tolist()
    file_index = {}
    
    print("Building static features index...")
    for filename, group in static_features_df.groupby('filename'):
        line_features = {}
        for _, row in group.iterrows():
            line_num = row['line_number']
            features = row[feature_cols].values.astype(np.float32)
            line_features[line_num] = features
        file_index[filename] = line_features
    
    print(f"Built index for {len(file_index)} files")
    return file_index, len(feature_cols)

def load_static_features_for_file_fast(file_index, feature_dim, filename, num_lines):
    """快速版本：使用预建索引加载静态特征"""
    if file_index is None:
        return np.zeros((num_lines, feature_dim), dtype=np.float32)
    
    features = []
    unmatched_count = 0
    
    # 获取该文件的特征映射
    file_features = file_index.get(filename, {})
    
    # 快速查找每一行的特征
    for line_idx in range(num_lines):
        line_num = line_idx + 1
        if line_num in file_features:
            features.append(file_features[line_num])
        else:
            features.append(np.zeros(feature_dim, dtype=np.float32))
            unmatched_count += 1
    
    # 只在未匹配行数较多时输出警告
    if unmatched_count > num_lines * 0.1:  # 超过10%的行未匹配才警告
        print(f"WARNING: File '{filename}' has {unmatched_count}/{num_lines} unmatched lines")
    
    return np.array(features)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def predict_defective_files_in_releases(args, dataset_name):
    actual_save_model_dir = args.save_model_dir + dataset_name + '/'

    train_rel = all_train_releases[dataset_name]
    test_rel = all_eval_releases[dataset_name][1:]

    # load the pre-trained CodeBERT model
    MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        codebert = model_class.from_pretrained(args.model_name_or_path,
                                               from_tf=bool('.ckpt' in args.model_name_or_path),
                                               config=config,
                                               cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        codebert = model_class(config)
  
    # 创建模型时传入静态特征相关参数
    model = BAFLineDP(
        embed_dim=args.embed_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        bafn_output_dim=args.bafn_hidden_dim,
        dropout=args.dropout,
        device=args.device,
        use_static_features=args.use_static_features,
        static_feature_dim=args.static_feature_dim
    )

    checkpoint = torch.load(actual_save_model_dir + 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    sig = nn.Sigmoid()

    codebert.to(args.device)
    model.to(args.device)
    model.eval()

    for rel in test_rel:
        print('generating prediction of release', rel, 'through', dataset_name)

        # 为每个测试版本加载对应的静态特征数据并预处理
        static_features_index = None
        feature_dim = 32  # 默认特征维度
        
        if args.use_static_features:
            static_feature_path = f"{args.static_features_dir}{rel}_ast_features.csv"
            if os.path.exists(static_feature_path):
                static_features_df = pd.read_csv(static_feature_path)
                static_features_df = load_and_process_static_features_for_prediction(static_features_df, rel)
                print(f"Loaded static features for {rel} from {static_feature_path}")
                
                # 预处理静态特征，建立快速查找索引
                static_features_index, feature_dim = preprocess_static_features(static_features_df)
            else:
                print(f"WARNING: Static features file not found for {rel}: {static_feature_path}")
                print("Proceeding without static features for this release")

        test_df = get_df(rel)
        row_list = []

        print(f"Processing {len(test_df.groupby('filename'))} files for {rel}...")
        
        for filename, df in tqdm(test_df.groupby('filename')):
            df = df[df['code_line'].ne('')]

            file_label = bool(df['file-label'].unique())
            line_label = df['line-label'].tolist()
            line_number = df['line_number'].tolist()
            is_comments = df['is_comment'].tolist()

            code = df['code_line'].tolist()

            # avoid memory overflow
            drop_length = 35000
            if len(code) > drop_length:
                continue

            code2d = prepare_code2d(code, True)
            code3d = [code2d]
            
            # 使用优化后的快速特征加载
            static_features = None
            if args.use_static_features and static_features_index is not None:
                static_feat = load_static_features_for_file_fast(static_features_index, feature_dim, filename, len(code))
                static_features = [static_feat]

            if args.use_static_features and static_features is not None:
                codevec = TextDataset(tokenizer, args, code3d, [file_label], static_features)
            else:
                codevec = TextDataset(tokenizer, args, code3d, [file_label])

            with torch.no_grad():
                input = torch.tensor(codevec.examples[0].input_ids)

                limit_length = 1000
                input = input.split(limit_length, 0)

                cov_input = [codebert(item.to(args.device), attention_mask=item.to(args.device).ne(1)).pooler_output
                             for item in input]
                cov_input = torch.cat(cov_input, dim=0)

                # 根据是否使用静态特征调用模型
                if args.use_static_features and codevec.examples[0].static_features is not None:
                    static_feat_tensor = torch.tensor(codevec.examples[0].static_features, dtype=torch.float32)
                    
                    # 处理静态特征的分块（与CodeBERT输出对应）
                    static_feat_chunks = []
                    start_idx = 0
                    for chunk in input:
                        chunk_size = chunk.shape[0]
                        end_idx = start_idx + chunk_size
                        if end_idx <= static_feat_tensor.shape[0]:
                            static_feat_chunks.append(static_feat_tensor[start_idx:end_idx])
                        else:
                            # 如果超出范围，用零填充
                            remaining_size = static_feat_tensor.shape[0] - start_idx
                            if remaining_size > 0:
                                last_chunk = static_feat_tensor[start_idx:]
                                padding = torch.zeros((chunk_size - remaining_size, static_feat_tensor.shape[1]))
                                static_feat_chunks.append(torch.cat([last_chunk, padding], dim=0))
                            else:
                                static_feat_chunks.append(torch.zeros((chunk_size, static_feat_tensor.shape[1])))
                        start_idx = end_idx
                    
                    # 如果静态特征块数量不够，用零填充
                    while len(static_feat_chunks) < len(input):
                        static_feat_chunks.append(torch.zeros((input[-1].shape[0], static_feat_tensor.shape[1])))
                    
                    static_feat_combined = torch.cat(static_feat_chunks, dim=0)[:cov_input.shape[0]]
                    
                    output, line_att_weight = model([cov_input], [static_feat_combined.to(args.device)])
                else:
                    output, line_att_weight = model([cov_input])
                
                file_prob = sig(output).item()
                prediction = bool(round(file_prob))

            torch.cuda.empty_cache()

            numpy_line_attn = line_att_weight[0].cpu().detach().numpy()

            for i in range(0, len(code)):
                cur_line = code[i]
                cur_line_label = line_label[i]
                cur_line_number = line_number[i]
                cur_is_comment = is_comments[i]
                cur_line_attn = numpy_line_attn[i] if i < len(numpy_line_attn) else 0.0
                

                row_dict = {
                    'project': dataset_name,
                    'train': train_rel,
                    'test': rel,
                    'filename': filename,
                    'file-level-ground-truth': file_label,
                    'prediction-prob': file_prob,
                    'prediction-label': prediction,
                    'line-number': cur_line_number,
                    'line-level-ground-truth': cur_line_label,
                    'is-comment-line': cur_is_comment,
                    'code-line': cur_line,
                    'line-attention-score': cur_line_attn
                }
                row_list.append(row_dict)

        df = pd.DataFrame(row_list)
        df.to_csv(args.prediction_dir + rel + '.csv', index=False)
        print('finished release', rel)

def main():
    arg = argparse.ArgumentParser()

    arg.add_argument('-file_lvl_gt', type=str, default='datasets/preprocessed_data/',
                     help='the directory of preprocessed data')
    arg.add_argument('-save_model_dir', type=str, default='output/model/BAFLineDP/',
                     help='the save directory of model')
    arg.add_argument('-prediction_dir', type=str, default='output/prediction/BAFLineDP/within-release/',
                     help='the results directory of prediction')

    # 新增静态特征相关参数
    arg.add_argument('-use_static_features', action='store_true', 
                     help='whether to use static features')
    arg.add_argument('-static_features_dir', type=str, 
                     default='datasets/manual_features/',
                     help='directory containing static features CSV files')
    arg.add_argument('-static_feature_dim', type=int, default=32,
                     help='dimension of static features')
    
    # 新增单项目预测参数
    arg.add_argument('-single_project', type=str, default=None,
                     help='predict on single project only (e.g., activemq)')

    arg.add_argument('-embed_dim', type=int, default=768, help='the input dimension of Bi-GRU')
    arg.add_argument('-gru_hidden_dim', type=int, default=64, help='hidden size of GRU')
    arg.add_argument('-gru_num_layers', type=int, default=1, help='number of GRU layer')
    arg.add_argument('-bafn_hidden_dim', type=int, default=256, help='output dimension of BAFN')
    arg.add_argument('-max_grad_norm', type=int, default=5, help='max gradient norm')
    arg.add_argument('-use_layer_norm', type=bool, default=True, help='weather to use layer normalization')
    arg.add_argument('-seed', type=int, default=0, help='random seed for initialization')
    arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')

    arg.add_argument('-model_type', type=str, default='roberta', help='the token embedding model')
    arg.add_argument('-model_name_or_path', type=str, default='models/codebert-base',
                     help='the model checkpoint for weights initialization')
    arg.add_argument('-config_name', type=str, default='models/codebert-base',
                     help='optional pretrained config name or path if not the same as model_name_or_path')
    arg.add_argument('-tokenizer_name', type=str, default='models/codebert-base',
                     help='optional pretrained tokenizer name or path if not the same as model_name_or_path')
    arg.add_argument('-cache_dir', type=str, default=None,
                     help='optional directory to store the pre-trained models')
    arg.add_argument('-block_size', type=int, default=75,
                     help='the training dataset will be truncated in block of this size for training')
    arg.add_argument('-do_lower_case', action='store_true', help='set this flag if you are using an uncased model')

    args = arg.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    if not os.path.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)

    # 如果指定了单个项目，只预测该项目；否则预测所有项目
    if args.single_project:
        if args.single_project in all_releases:
            print(f"Predicting on single project: {args.single_project}")
            predict_defective_files_in_releases(args, args.single_project)
        else:
            print(f"Project {args.single_project} not found in all_releases")
            print(f"Available projects: {list(all_releases.keys())}")
    else:
        print("Predicting on all projects")
        dataset_names = list(all_releases.keys())
        for dataset_name in dataset_names:
            predict_defective_files_in_releases(args, dataset_name)

if __name__ == "__main__":
    main()
    
    
    
    
    