"""
Same as DataLoader_4class_patient_v2.py (edge_dim=4 fused FC edges), plus a
`fc_flat` attribute per sample: the same flattened+concatenated FC vector
(1444-dim: 4 fc_types x 19x19) that the CNN/MLP/SVM baselines use directly as
input. GATv2Conv (v3) and focal loss (v4) both failed to close GAT's gap with
the baselines, and both failures were concentrated on LPD -- suggesting GAT's
edge-attention channel alone doesn't carry enough signal to separate LPD from
OTHERS. This loader lets a hybrid model access the same full-resolution FC
representation the baselines have, in parallel with the graph structure.

`fc_flat` is stored with an extra leading dim of size 1 (shape (1, 1444))
so that PyG's default batching (concatenation along dim 0 across the
graphs in a batch) produces the correct (batch_size, 1444) shape -- without
the leading dim, concatenation would flatten all samples into one 1D vector.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class GraphLoader(Dataset):
    def __init__(self, psd_folder, fc_folder, fc_types=['corr', 'coh', 'aec', 'plv'], label_file='train.csv',
                 random_seed=42):
        super(GraphLoader, self).__init__()
        self.psd_folder = psd_folder
        self.fc_folder = fc_folder
        self.fc_types = fc_types
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.label_map = {'SEIZURE': 0, 'LPD': 1, 'GPD': 2, 'OTHERS': 3}

        self.labels_df = pd.read_csv(label_file)
        self.labels_df['file_name'] = self.labels_df['eeg_id'].astype(str) + '_' + self.labels_df['eeg_sub_id'].astype(
            str) + '.npy'
        self.file_to_patient = dict(zip(self.labels_df['file_name'], self.labels_df['patient_id']))

        self.file_names = self._select_balanced_files()

        self.psd_scaler = StandardScaler()
        self.fit_psd_scaler()

    def _select_balanced_files(self):
        selected_files = []

        existing_files = set([f for f in os.listdir(self.psd_folder) if f.endswith('.npy')])
        valid_files_df = self.labels_df[self.labels_df['file_name'].isin(existing_files)].copy()

        print("\n기본 클래스 샘플링:")
        for label in tqdm(['SEIZURE', 'LPD', 'GPD'], desc="Sampling major classes"):
            label_files = valid_files_df[valid_files_df['expert_consensus'].str.upper() == label]['file_name'].tolist()
            if len(label_files) < 9000:
                print(f"\n경고: {label} 클래스의 파일 수가 9000개 미만입니다. (found: {len(label_files)})")
            selected = np.random.choice(label_files, size=min(9000, len(label_files)), replace=False)
            selected_files.extend(selected)

        print("\nOTHERS 클래스 샘플링:")
        for label in tqdm(['LRDA', 'GRDA', 'OTHER'], desc="Sampling OTHERS class"):
            label_files = valid_files_df[valid_files_df['expert_consensus'].str.upper() == label]['file_name'].tolist()
            if len(label_files) < 3000:
                print(f"\n경고: {label} 클래스의 파일 수가 3000개 미만입니다. (found: {len(label_files)})")
            selected = np.random.choice(label_files, size=min(3000, len(label_files)), replace=False)
            selected_files.extend(selected)

        print(f"\n총 {len(selected_files)}개 파일 선택 완료")
        return selected_files

    def fit_psd_scaler(self):
        print("\nPSD 스케일러 피팅:")
        all_psd_data = []
        for file_name in tqdm(self.file_names, desc="Loading PSD data for scaling"):
            psd_path = os.path.join(self.psd_folder, file_name)
            psd_data = np.load(psd_path)
            all_psd_data.append(psd_data)
        all_psd_data = np.concatenate(all_psd_data, axis=0)
        self.psd_scaler.fit(all_psd_data)
        print("스케일러 피팅 완료")

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        file_name = self.file_names[idx]
        psd_path = os.path.join(self.psd_folder, file_name)
        node_features = self.psd_scaler.transform(np.load(psd_path))
        node_features = torch.FloatTensor(node_features)
        edge_index, edge_attr, fc_flat = self.load_fc_data(file_name)
        label = self.extract_label(file_name)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr,
                    fc_flat=fc_flat, y=label)

    def load_fc_data(self, file_name):
        """One edge per (ordered) channel pair, edge_attr = [corr, coh, aec, plv]
        at that pair -- fills 0 for any fc_type file missing for this sample.
        Also returns fc_flat: the same 4 matrices flattened+concatenated
        (shape (1, n_types*n_nodes*n_nodes)), for the hybrid model's flat branch."""
        n_types = len(self.fc_types)
        matrices = []
        n_nodes = None
        for fc_type in self.fc_types:
            fc_path = os.path.join(self.fc_folder, fc_type, file_name)
            if os.path.exists(fc_path):
                m = np.load(fc_path)
                matrices.append(m)
                n_nodes = m.shape[0]
            else:
                matrices.append(None)

        if n_nodes is None:
            return (torch.LongTensor([[], []]), torch.FloatTensor(np.zeros((0, n_types))),
                    torch.zeros((1, 0)))

        matrices = [m if m is not None else np.zeros((n_nodes, n_nodes)) for m in matrices]

        edge_index = []
        edge_attr = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([m[i, j] for m in matrices])
                edge_attr.append([m[j, i] for m in matrices])

        fc_flat = np.concatenate([m.flatten() for m in matrices])
        fc_flat = torch.FloatTensor(fc_flat).unsqueeze(0)

        return torch.LongTensor(edge_index).t().contiguous(), torch.FloatTensor(edge_attr), fc_flat

    def extract_label(self, file_name):
        label = self.labels_df[self.labels_df['file_name'] == file_name]['expert_consensus'].iloc[0].upper()
        if label in ['LRDA', 'GRDA', 'OTHER']:
            return torch.LongTensor([self.label_map['OTHERS']])
        elif label in self.label_map:
            return torch.LongTensor([self.label_map[label]])
        return None

    @staticmethod
    def create_data_loaders(psd_folders, fc_folders, fc_types=['corr', 'coh', 'aec', 'plv'], batch_size=32,
                            label_file='train.csv', random_state=42,
                            save_split=True, split_save_path='fold_splits_patient.json'):
        """
        Patient-grouped 10-fold CV: no patient's segments appear in more than one
        of {train, val, test} within a fold. Uses StratifiedGroupKFold (outer,
        test split) + StratifiedGroupKFold (inner, val split from train_val),
        grouped by patient_id.
        """
        combined_dataset = GraphLoader.create_combined_dataset(psd_folders, fc_folders, fc_types, label_file,
                                                               random_state)

        def get_file_name(global_idx):
            dataset_idx, sample_idx = 0, global_idx
            cumulative = 0
            for ds in combined_dataset.datasets:
                if sample_idx < cumulative + len(ds):
                    sample_idx = sample_idx - cumulative
                    real_idx = ds.indices[sample_idx]
                    return ds.dataset.file_names[real_idx]
                cumulative += len(ds)
            return None

        base_loader = combined_dataset.datasets[0].dataset
        label_str_lookup = dict(zip(base_loader.labels_df['file_name'],
                                     base_loader.labels_df['expert_consensus'].str.upper()))
        label_map = base_loader.label_map

        def label_str_to_int(s):
            return label_map['OTHERS'] if s in ['LRDA', 'GRDA', 'OTHER'] else label_map[s]

        def get_label_str(file_name, global_idx):
            return label_str_lookup.get(file_name)

        all_files = [get_file_name(i) for i in range(len(combined_dataset))]
        all_labels = np.array([label_str_to_int(label_str_lookup[f]) for f in all_files])

        file_to_patient = combined_dataset.datasets[0].dataset.file_to_patient
        all_patients = np.array([file_to_patient[f] for f in all_files])

        outer_kf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)

        fold_datasets = []
        fold_split_info = []

        for fold, (train_val_idx, test_idx) in enumerate(
                outer_kf.split(np.zeros(len(all_labels)), all_labels, groups=all_patients)):

            test_dataset = torch.utils.data.Subset(combined_dataset, test_idx)

            train_val_labels = all_labels[train_val_idx]
            train_val_patients = all_patients[train_val_idx]

            inner_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state + fold)
            inner_train_idx, inner_val_idx = next(
                inner_kf.split(np.zeros(len(train_val_idx)), train_val_labels, groups=train_val_patients)
            )

            final_train_idx = train_val_idx[inner_train_idx]
            final_val_idx = train_val_idx[inner_val_idx]

            p_train = set(all_patients[final_train_idx])
            p_val = set(all_patients[final_val_idx])
            p_test = set(all_patients[test_idx])
            assert not (p_train & p_val), f"fold {fold}: train/val patient overlap"
            assert not (p_train & p_test), f"fold {fold}: train/test patient overlap"
            assert not (p_val & p_test), f"fold {fold}: val/test patient overlap"

            train_dataset = torch.utils.data.Subset(combined_dataset, final_train_idx)
            val_dataset = torch.utils.data.Subset(combined_dataset, final_val_idx)

            loader_kwargs = dict(num_workers=12, pin_memory=True, persistent_workers=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

            fold_datasets.append((train_loader, val_loader, test_loader))

            if save_split:
                def collect_split_info(indices, split_name):
                    files, labels_int, labels_str, patients = [], [], [], []
                    for idx in indices:
                        fname = get_file_name(idx)
                        label_int = int(all_labels[idx])
                        label_raw = get_label_str(fname, idx)
                        files.append(fname)
                        labels_int.append(label_int)
                        labels_str.append(label_raw)
                        patients.append(int(file_to_patient[fname]))
                    return {'files': files, 'labels_int': labels_int, 'labels_str': labels_str, 'patient_id': patients}

                fold_info = {
                    'fold': fold,
                    'train': collect_split_info(final_train_idx, 'train'),
                    'val':   collect_split_info(final_val_idx,   'val'),
                    'test':  collect_split_info(test_idx,        'test'),
                    'n_unique_patients_train': len(p_train),
                    'n_unique_patients_val': len(p_val),
                    'n_unique_patients_test': len(p_test),
                }
                fold_split_info.append(fold_info)
                print(f"Fold {fold+1}: train={len(final_train_idx)} ({len(p_train)} patients), "
                      f"val={len(final_val_idx)} ({len(p_val)} patients), "
                      f"test={len(test_idx)} ({len(p_test)} patients)  [patient-disjoint OK]")

        if save_split:
            with open(split_save_path, 'w') as f:
                json.dump(fold_split_info, f, indent=2)
            print(f"\nFold split 저장 완료: {split_save_path}")

        return fold_datasets

    @staticmethod
    def load_fold_splits(split_path='fold_splits_patient.json'):
        with open(split_path, 'r') as f:
            fold_split_info = json.load(f)
        print(f"Fold split 로드 완료: {split_path} ({len(fold_split_info)} folds)")
        return fold_split_info

    @staticmethod
    def create_combined_dataset(psd_folders, fc_folders, fc_types=['corr', 'coh', 'aec', 'plv'],
                                label_file='train.csv', random_state=42):
        datasets = []
        for psd_folder, fc_folder in tqdm(zip(psd_folders, fc_folders), desc="Loading datasets"):
            dataset = GraphLoader(psd_folder, fc_folder, fc_types, label_file, random_state)
            label_lookup = dict(zip(dataset.labels_df['file_name'],
                                     dataset.labels_df['expert_consensus'].str.upper()))
            valid_labels = set(dataset.label_map.keys()) | {'LRDA', 'GRDA', 'OTHER'}
            valid_indices = [i for i, fn in enumerate(dataset.file_names)
                              if label_lookup.get(fn) in valid_labels]
            datasets.append(torch.utils.data.Subset(dataset, valid_indices))

        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        return combined_dataset
