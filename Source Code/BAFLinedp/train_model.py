import copy
import os, argparse
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
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


def load_and_process_static_features(static_features_df, dataset_name):
    if static_features_df is None:
        return None, None
    
    feature_cols = static_features_df.columns[4:].tolist()
    
    feature_data = static_features_df[feature_cols].values
    
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(feature_data)
    
    static_features_df = static_features_df.copy()
    static_features_df[feature_cols] = normalized_features
    
    print(f"Static features for {dataset_name}:")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Feature range after normalization: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")
    
    return static_features_df, scaler


def load_static_features_for_file(static_features_df, filename, code_lines, is_training=True):
    if static_features_df is None:
        return np.zeros((len(code_lines), 32), dtype=np.float32)
    
    feature_cols = static_features_df.columns[4:].tolist()
    
    file_features = static_features_df[static_features_df['filename'] == filename]
    
    if len(file_features) == 0:
        print(f"WARNING: No features found for file {filename}, using zero vectors")
        return np.zeros((len(code_lines), len(feature_cols)), dtype=np.float32)
    
    features = []
    unmatched_lines = []
    
    for line_idx in range(len(code_lines)):
        matching_rows = file_features[file_features['line_number'] == line_idx + 1]
        
        if len(matching_rows) > 0:
            static_feature = matching_rows[feature_cols].values[0].astype(np.float32)
            features.append(static_feature)
        else:
            unmatched_lines.append(line_idx + 1)
            static_feature = np.zeros(len(feature_cols), dtype=np.float32)
            features.append(static_feature)
    
    if unmatched_lines and is_training:
        mode = "training" if is_training else "prediction"
        if len(unmatched_lines) > 10: 
            print(f"WARNING during {mode}: File '{filename}' has {len(unmatched_lines)} unmatched lines")
            print(f"  Total lines in file: {len(code_lines)}")
            print(f"  Using zero vectors for unmatched lines")
    
    return np.array(features)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    file_data = [data_list for data_list in batch]
    return file_data


def get_loss_weight(labels, weight_dict):
    label_list = labels.numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])

    weight_tensor = torch.tensor(weight_list).reshape(-1, 1)

    return weight_tensor


def get_code3d_and_label_with_static(df, static_features_df, to_lowercase=False, max_sent_length=None):
    code3d = []
    all_file_label = []
    static_features_list = []
    
    print(f"Processing {len(df.groupby('filename'))} files...")
    
    for i, (filename, group_df) in enumerate(df.groupby('filename')):
        if i % 100 == 0: 
            print(f"Processing file {i+1}/{len(df.groupby('filename'))}: {filename}")
        
        group_df = group_df[group_df['code_line'].ne('')]
        file_label = bool(group_df['file-label'].unique())
        code = list(group_df['code_line'])
        
        if max_sent_length:
            code2d = prepare_code2d(code, to_lowercase)[:max_sent_length]
            code = code[:max_sent_length]  
            code2d = prepare_code2d(code, to_lowercase)
        
        if static_features_df is not None:
            static_features = load_static_features_for_file(static_features_df, filename, code, is_training=True)
            static_features_list.append(static_features)
        else:
            static_features_list.append(None)
        
        code3d.append(code2d)
        all_file_label.append(file_label)
    
    print(f"Finished processing all {len(code3d)} files")
    return code3d, all_file_label, static_features_list


def train_model(args, dataset_name):
    actual_save_model_dir = args.save_model_dir + dataset_name + '/'

    if not os.path.exists(actual_save_model_dir):
        os.makedirs(actual_save_model_dir)

    if not os.path.exists(args.loss_dir):
        os.makedirs(args.loss_dir)

    train_rel = all_train_releases[dataset_name]
    valid_rel = all_eval_releases[dataset_name][0]

    train_df = get_df(train_rel)
    valid_df = get_df(valid_rel)

    static_features_df = None
    if args.use_static_features:
        train_version = all_train_releases[dataset_name]
        static_feature_path = f"{args.static_features_dir}{train_version}_ast_features.csv"
        if os.path.exists(static_feature_path):
            static_features_df = pd.read_csv(static_feature_path)
            static_features_df, scaler = load_and_process_static_features(static_features_df, dataset_name)
            print(f"Loaded and normalized static features from {static_feature_path}")
        else:
            print(f"ERROR: Static features file not found: {static_feature_path}")
            print(f"Expected file: {train_version}_ast_features.csv in {args.static_features_dir}")
            return

    print("Loading training data...")
    if args.use_static_features:
        train_code3d, train_label, train_static_features = get_code3d_and_label_with_static(
            train_df, static_features_df, True, args.max_train_LOC)
        
        print("Loading validation data...")
        valid_version = all_eval_releases[dataset_name][0]
        valid_static_feature_path = f"{args.static_features_dir}{valid_version}_ast_features.csv"
        valid_static_features_df = None
        
        if os.path.exists(valid_static_feature_path):
            valid_static_features_df = pd.read_csv(valid_static_feature_path)
            valid_static_features_df, _ = load_and_process_static_features(valid_static_features_df, f"{dataset_name}_valid")
            print(f"Loaded validation static features from {valid_static_feature_path}")
        else:
            print(f"WARNING: Validation static features file not found: {valid_static_feature_path}")
            print("Using training static features for validation")
            valid_static_features_df = static_features_df
        
        valid_code3d, valid_label, valid_static_features = get_code3d_and_label_with_static(
            valid_df, valid_static_features_df, True, args.max_train_LOC)
    else:
        train_code3d, train_label = get_code3d_and_label(train_df, True, args.max_train_LOC)
        valid_code3d, valid_label = get_code3d_and_label(valid_df, True, args.max_train_LOC)
        train_static_features = None
        valid_static_features = None

    print(f"Loaded {len(train_code3d)} training files and {len(valid_code3d)} validation files")

    # apply weighted loss to handle class imbalance
    sample_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)

    weight_dict = {}
    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)

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

    codebert.to(args.device)
    model.to(args.device)

    # convert input format as required by CodeBERT
    if args.use_static_features:
        x_train_vec = TextDataset(tokenizer, args, train_code3d, train_label, train_static_features)
        x_valid_vec = TextDataset(tokenizer, args, valid_code3d, valid_label, valid_static_features)
    else:
        x_train_vec = TextDataset(tokenizer, args, train_code3d, train_label)
        x_valid_vec = TextDataset(tokenizer, args, valid_code3d, valid_label)

    train_dl = DataLoader(x_train_vec, shuffle=True, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn)
    valid_dl = DataLoader(x_valid_vec, shuffle=False, batch_size=args.batch_size, drop_last=False,
                          collate_fn=collate_fn)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    sig = nn.Sigmoid()

    best_auc = 0
    best_epoch = 0
    best_model = None

    train_loss_all_epochs = []
    val_loss_all_epochs = []
    val_auc_all_epochs = []

    model.zero_grad()
    for epoch in range(1, args.num_epochs + 1):
        train_losses = []
        val_losses = []

        # training model
        model.train()
        for step, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc='Train Loop'):
            if args.use_static_features:
                inputs = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                static_features = [item[2] for item in batch]
            else:
                inputs = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                static_features = None

            labels = torch.tensor(labels)

            # initial acquisition of code line semantics
            cov_inputs = []
            with torch.no_grad():
                for item in inputs:
                    cov_inputs.append(
                        codebert(item.to(args.device), attention_mask=item.to(args.device).ne(1)).pooler_output
                    )

            weight_tensor = get_loss_weight(labels, weight_dict)
            criterion.weight = weight_tensor.to(args.device)

            if args.use_static_features:
                static_features_tensors = [sf.to(args.device) for sf in static_features]
                output, _ = model(cov_inputs, static_features_tensors)
            else:
                output, _ = model(cov_inputs)
                
            loss = criterion(output, labels.reshape(args.batch_size, 1).to(args.device))
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # gradient clipping to prevent gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            torch.cuda.empty_cache()

        train_loss_all_epochs.append(np.mean(train_losses))
        outputs = []
        outputs_labels = []

        # evaluating model
        with torch.no_grad():
            criterion.weight = None
            model.eval()

            for step, batch in tqdm(enumerate(valid_dl), total=len(valid_dl), desc='Valid Loop'):
                if args.use_static_features:
                    inputs = [item[0] for item in batch]
                    labels = [item[1] for item in batch]
                    static_features = [item[2] for item in batch]
                else:
                    inputs = [item[0] for item in batch]
                    labels = [item[1] for item in batch]
                    static_features = None

                labels = torch.tensor(labels)

                cov_inputs = []
                for item in inputs:
                    cov_inputs.append(
                        codebert(item.to(args.device), attention_mask=item.to(args.device).ne(1)).pooler_output
                    )

                if args.use_static_features:
                    static_features_tensors = [sf.to(args.device) for sf in static_features]
                    output, _ = model(cov_inputs, static_features_tensors)
                else:
                    output, _ = model(cov_inputs)

                outputs.append(sig(output))
                outputs_labels.append(labels)

                val_loss = criterion(output, labels.reshape(len(labels), 1).to(args.device))
                val_losses.append(val_loss.item())

        val_loss_all_epochs.append(np.mean(val_losses))

        # compute the metric of AUC
        y_prob = torch.cat(outputs)
        y_gt = torch.cat(outputs_labels)

        valid_auc = roc_auc_score(y_gt, y_prob.to('cpu'))
        val_auc_all_epochs.append(valid_auc)

        if valid_auc >= best_auc:
            best_model = copy.deepcopy(model)
            best_auc = valid_auc
            best_epoch = epoch

        print('Training at Epoch ' + str(epoch) + ' with training loss ' + str(np.mean(train_losses)))
        print('Validation at Epoch ' + str(epoch) + ' with validation loss ' + str(np.mean(val_losses)),
              ' AUC ' + str(valid_auc))

        # save the best model
        if epoch % args.num_epochs == 0:
            print('The training step of ' + dataset_name + ' is finished!')

            torch.save({'epoch': best_epoch,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       actual_save_model_dir + 'best_model.pth')

        loss_df = pd.DataFrame()
        loss_df['epoch'] = np.arange(1, len(train_loss_all_epochs) + 1)
        loss_df['train_loss'] = train_loss_all_epochs
        loss_df['valid_loss'] = val_loss_all_epochs
        loss_df['valid_auc'] = val_auc_all_epochs

        loss_df.to_csv(args.loss_dir + dataset_name + '-loss_record.csv', index=False)


def main():
    arg = argparse.ArgumentParser()

    arg.add_argument('-file_lvl_gt', type=str, default='datasets/preprocessed_data/',
                     help='the directory of preprocessed data')
    arg.add_argument('-save_model_dir', type=str, default='output/model/BAFLineDP/',
                     help='the save directory of model')
    arg.add_argument('-loss_dir', type=str, default='output/loss/BAFLineDP/',
                     help='the loss directory of model')

    arg.add_argument('-use_static_features', action='store_true', 
                     help='whether to use static features')
    arg.add_argument('-static_features_dir', type=str, 
                     default='datasets/manual_features/',
                     help='directory containing static features CSV files')
    arg.add_argument('-static_feature_dim', type=int, default=32,
                     help='dimension of static features')

    arg.add_argument('-single_project', type=str, default=None,
                     help='train on single project only (e.g., activemq)')

    arg.add_argument('-batch_size', type=int, default=16, help='batch size per GPU/CPU for training/evaluation')
    arg.add_argument('-num_epochs', type=int, default=10, help='total number of training epochs to perform')
    arg.add_argument('-embed_dim', type=int, default=768, help='the input dimension of Bi-GRU')
    arg.add_argument('-gru_hidden_dim', type=int, default=64, help='hidden size of GRU')
    arg.add_argument('-gru_num_layers', type=int, default=1, help='number of GRU layer')
    arg.add_argument('-bafn_hidden_dim', type=int, default=256, help='output dimension of BAFN')
    arg.add_argument('-max_grad_norm', type=int, default=5, help='max gradient norm')
    arg.add_argument('-max_train_LOC', type=int, default=900, help='max LOC of training/validation data')
    arg.add_argument('-use_layer_norm', type=bool, default=True, help='weather to use layer normalization')
    arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')
    arg.add_argument('-lr', type=float, default=0.001, help='learning rate')
    arg.add_argument('-seed', type=int, default=0, help='random seed for initialization')
    arg.add_argument('-weight_decay', type=float, default=0.0, help='weight decay whether apply some')

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

    if args.single_project:
        if args.single_project in all_releases:
            print(f"Training on single project: {args.single_project}")
            train_model(args, args.single_project)
        else:
            print(f"Project {args.single_project} not found in all_releases")
            print(f"Available projects: {list(all_releases.keys())}")
    else:
        print("Training on all projects")
        dataset_names = list(all_releases.keys())
        for dataset_name in dataset_names:
            train_model(args, dataset_name)


if __name__ == "__main__":
    main()
    
    