import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast # AMP için eklendi
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder # LabelEncoder'ı import et
import joblib # joblib'i import et
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt

# ---------------------------
# 1. Reproducibility
# ---------------------------
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# 2. Hyperparameters
# ---------------------------
skiprows            = 28
wl_min, wl_max      = 950, 1650
epochs              = 100
early_stop_patience = 15
batch_size          = 32
validation_split    = 0.3
learning_rate       = 0.001
criterion           = nn.CrossEntropyLoss()

# ---------------------------
# 3. Preprocessing functions
# ---------------------------
def derivative(input_data):
    data_der = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        data_der[i, :] = np.gradient(input_data[i, :])
    return data_der


def det(input_data, order=2):
    data_det = np.zeros_like(input_data)
    x = np.arange(input_data.shape[1])
    for i in range(input_data.shape[0]):
        coeffs = np.polyfit(x, input_data[i, :], order)
        baseline = np.polyval(coeffs, x)
        data_det[i, :] = input_data[i, :] - baseline
    return data_det


def snv(input_data):
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        mean = np.mean(input_data[i, :])
        std  = np.std(input_data[i, :])
        data_snv[i, :] = (input_data[i, :] - mean) / std
    return data_snv

# ---------------------------
# 4. Dataset class
# ---------------------------
class SpectralDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

# ---------------------------
# 5. Load data utility
# ---------------------------
def load_data(data_dir):
    X, y, paths = [], [], []
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    if not classes: raise ValueError(f"No classes in {data_dir}")
    label_map = {c: i for i, c in enumerate(classes)}
    for c in classes:
        cls_folder = os.path.join(data_dir, c)
        for sample_subfolder_name in os.listdir(cls_folder):
            sample_folder_path = os.path.join(cls_folder, sample_subfolder_name)
            if not os.path.isdir(sample_folder_path): continue
            for f_name in os.listdir(sample_folder_path):
                if f_name.lower().endswith('_a.csv'):
                    file_full_path = os.path.join(sample_folder_path, f_name)
                    try:
                        df = pd.read_csv(file_full_path, skiprows=skiprows)
                        if df.shape[1] < 2:
                            print(f"Warning: File {file_full_path} has less than 2 columns. Skipping.")
                            continue

                        wl = df.iloc[:,0].values
                        # Explicitly convert absorbance column to numeric, coercing errors to NaN
                        ab_series = pd.to_numeric(df.iloc[:,1], errors='coerce')

                        if ab_series.isnull().any():
                            print(f"Warning: File {file_full_path} contains non-numeric absorbance values. These were converted to NaN.")
                        
                        ab = ab_series.values # Now 'ab' is a float64 array, possibly with NaNs

                    except Exception as e:
                        print(f"Error reading or processing file {file_full_path}: {e}. Skipping.")
                        continue

                    mask = (wl>=wl_min)&(wl<=wl_max)
                    current_spectrum = ab[mask]

                    if np.isnan(current_spectrum).any():
                        print(f"Warning: Spectrum from {file_full_path} contains NaN values after wavelength filtering. This might affect polyfit. Skipping this file.")
                        continue # Skip files with NaNs in the relevant spectral region

                    if not current_spectrum.size > 0: # Ensure there's data after masking
                        print(f"Warning: No data in wavelength range {wl_min}-{wl_max} for {file_full_path} or spectrum is empty after masking. Skipping.")
                        continue

                    X.append(current_spectrum)
                    y.append(label_map[c])
                    paths.append((c, sample_subfolder_name, f_name)) # Store (class, subfolder, filename)
    if not X: raise ValueError("No spectra found.")

    # Ensure all spectra have the same length after masking
    feature_lengths = [len(x_spec) for x_spec in X]
    expected_length = feature_lengths[0] if feature_lengths else 0
    if not all(l == expected_length for l in feature_lengths):
        for i, length in enumerate(feature_lengths):
            if length != expected_length:
                print(f"  ERROR: Inconsistent length ({length}, expected {expected_length}) found for file: {paths[i][0]}/{paths[i][1]}/{paths[i][2]}")
        raise ValueError("Spectra have inconsistent lengths after wavelength filtering. Check logs for problematic files.")

    return np.array(X, dtype=np.float64), np.array(y), paths, classes # Use np.array and specify dtype

# ---------------------------
# 6. Model definition
# ---------------------------
class CNN_CuiFearn_Adapted(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        k1, k2, c1, c2 = 10,10,16,32
        self.conv1 = nn.Conv1d(1,c1,k1)
        self.conv2 = nn.Conv1d(c1,c2,k2)
        conv_out = input_size - k1 - k2 + 2
        self.fc1 = nn.Linear(c2*conv_out,128)
        self.fc2 = nn.Linear(128,n_classes)
        self.shapes_printed_once = False # Bayrağı burada ekliyoruz

    def forward(self,x):
        if not self.shapes_printed_once:
            print(f"  Forward pass (first time) - Input shape: {x.shape}")
            x_conv1 = F.relu(self.conv1(x))
            print(f"  Forward pass (first time) - Shape after conv1 + ReLU: {x_conv1.shape}")
            x_conv2 = F.relu(self.conv2(x_conv1))
            print(f"  Forward pass (first time) - Shape after conv2 + ReLU: {x_conv2.shape}")
            x_flat = x_conv2.view(x_conv2.size(0),-1)
            print(f"  Forward pass (first time) - Shape after flatten: {x_flat.shape}")
            x_fc1 = F.relu(self.fc1(x_flat))
            print(f"  Forward pass (first time) - Shape after fc1 + ReLU: {x_fc1.shape}")
            x_out = self.fc2(x_fc1)
            print(f"  Forward pass (first time) - Shape after fc2 (output): {x_out.shape}")
            self.shapes_printed_once = True # Bayrağı True yapıyoruz
            return x_out
        else: # Şekiller zaten yazdırıldıysa, normal forward pass
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0),-1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x

# ---------------------------
# Helper functions for single spectrum loading and processing (for plotting misclassified)
# ---------------------------
def load_single_spectrum(file_path, min_wl, max_wl, skiprows_val):
    """Loads a single spectrum file and filters by wavelength."""
    try:
        df = pd.read_csv(file_path, skiprows=skiprows_val)
        wl, ab = df.iloc[:,0].values, df.iloc[:,1].values
        mask = (wl >= min_wl) & (wl <= max_wl)
        if not np.any(mask): # Check if any data remains after filtering
            print(f"Warning: No data in wavelength range {min_wl}-{max_wl} for {file_path}")
            return np.array([]), np.array([])
        return wl[mask], ab[mask]
    except Exception as e:
        print(f"Error loading single spectrum {file_path}: {e}")
        return np.array([]), np.array([])

def preprocess_single_spectrum(spectrum_data):
    """Applies preprocessing (derivative, det, snv) to a single spectrum (1D array)."""
    if spectrum_data.ndim == 1:
        spectrum_data = spectrum_data.reshape(1, -1)  # Needs to be 2D for preprocessing functions
    processed_data = snv(derivative(det(spectrum_data))) # İstenen Sıra: det -> derivative -> snv
    return processed_data.flatten() # Return as 1D

# ---------------------------
# 7. Main pipeline
# ---------------------------
def main():
    train_dir = '/home/han/Documents/INNO_MAIN_1000_Train_200_Test_After_KUTU_LDPE_ADDED/Train'
    test_dir  = '/home/han/Documents/INNO_MAIN_1000_Train_200_Test_After_KUTU_LDPE_ADDED/Prediction'
    target_save_dir = "/home/han/Documents/Best_Results" # Yeni hedef kaydetme dizini

    # 7.1 Load raw data
    X_raw, y_raw, paths_raw, classes = load_data(train_dir)
    X_test_raw, y_test, test_paths, _ = load_data(test_dir)

    # LabelEncoder'ı oluştur, eğit ve kaydet
    # classes listesi load_data'dan sıralı sınıf isimlerini içerir
    label_encoder = LabelEncoder()
    label_encoder.fit(classes) # classes listesi ile eğit
    encoder_save_path = os.path.join(target_save_dir, "label_encoder.pkl")
    joblib.dump(label_encoder, encoder_save_path)
    print(f"Label encoder saved to {encoder_save_path}")

    # 7.2 Train/Validation Split: Samples within each subfolder are split
    # --- New Split Logic ---
    # 1. Group sample indices by their subfolder
    #    subfolder_id is "ClassName/SubfolderName"
    subfolder_to_indices_map = {}
    for i, (cls_name, sub_name, _) in enumerate(paths_raw):
        subfolder_key = f"{cls_name}/{sub_name}"
        if subfolder_key not in subfolder_to_indices_map:
            subfolder_to_indices_map[subfolder_key] = []
        subfolder_to_indices_map[subfolder_key].append(i)

    final_train_indices = []
    final_val_indices = []

    # For printing distribution later
    train_samples_per_subfolder = {key: 0 for key in subfolder_to_indices_map}
    val_samples_per_subfolder = {key: 0 for key in subfolder_to_indices_map}

    for subfolder_key, original_indices_in_subfolder in subfolder_to_indices_map.items():
        num_samples = len(original_indices_in_subfolder)

        # Shuffle indices for this subfolder
        shuffled_subfolder_indices = list(original_indices_in_subfolder) # Make a mutable copy
        random.shuffle(shuffled_subfolder_indices)

        if num_samples == 0:
            continue
        
        if num_samples == 1:
            # If only one sample, assign it to training
            final_train_indices.extend(shuffled_subfolder_indices)
            train_samples_per_subfolder[subfolder_key] += len(shuffled_subfolder_indices)
        else:
            # Calculate number of validation samples
            n_val = int(round(num_samples * validation_split))
            
            # Ensure at least one sample for validation if validation_split > 0 and possible
            if n_val == 0 and validation_split > 0.0 and num_samples > 1:
                n_val = 1
                
            # Ensure at least one sample for training if possible
            if n_val == num_samples and num_samples > 0:
                n_val = num_samples - 1
            
            n_val = max(0, n_val) # Ensure n_val is not negative

            val_idx_for_subfolder = shuffled_subfolder_indices[:n_val]
            train_idx_for_subfolder = shuffled_subfolder_indices[n_val:]
            
            final_val_indices.extend(val_idx_for_subfolder)
            final_train_indices.extend(train_idx_for_subfolder)

            val_samples_per_subfolder[subfolder_key] += len(val_idx_for_subfolder)
            train_samples_per_subfolder[subfolder_key] += len(train_idx_for_subfolder)

    train_idx = np.array(list(set(final_train_indices))) # Use set to remove duplicates if any, then list, then np.array
    val_idx = np.array(list(set(final_val_indices)))
    # --- End of Custom Split Logic ---

    X_tr_raw, y_tr = X_raw[train_idx], y_raw[train_idx]
    X_val_raw, y_val = X_raw[val_idx], y_raw[val_idx]
    paths_tr = [paths_raw[i] for i in train_idx]
    paths_val = [paths_raw[i] for i in val_idx]

    # 7.3 Verify no sample-level overlap (specific _a.csv files)
    # paths_tr and paths_val now contain (class, subfolder, filename) tuples
    leaked_files_train_val = set(paths_tr) & set(paths_val)
    if leaked_files_train_val:
        print("\nLeak detected between TRAIN and VAL samples (specific _a.csv files):")
        for c, s, f_name in sorted(list(leaked_files_train_val)):
            print(f"  File: {c}/{s}/{f_name} found in both train and val sets.")
    else:
        print("No leakage of specific _a.csv files between train and val samples.")

    # 7.3.1 Verify no sample-level overlap between train and prediction (specific _a.csv files)
    leaked_files_train_test = set(paths_tr) & set(test_paths)
    if leaked_files_train_test:
        print("\nLeak detected between TRAIN and PREDICTION samples (specific _a.csv files):")
        for c, s, f_name in sorted(list(leaked_files_train_test)):
            print(f"  File: {c}/{s}/{f_name} found in both train and prediction sets.")
    else:
        print("No leakage of specific _a.csv files between train and prediction samples.")

    # 7.3.2 Verify no sample-level overlap between combined train/val (raw data) and prediction (specific _a.csv files)
    # paths_raw contains all (class, subfolder, filename) tuples from train_dir
    leaked_files_raw_test = set(paths_raw) & set(test_paths)
    if leaked_files_raw_test:
        print("\nLeak detected between COMBINED TRAIN/VAL source data and PREDICTION samples (specific _a.csv files):")
        for c, s, f_name in sorted(list(leaked_files_raw_test)):
            print(f"  File: {c}/{s}/{f_name} found in both combined train/val source and prediction sets.")
    else:
        print("No leakage of specific _a.csv files between combined train/val source data and prediction samples.")
    
    # Calculate sample counts per main class for the final train/val sets
    if y_tr.size > 0:
        train_counts_per_main_class_id = np.bincount(y_tr.astype(int), minlength=len(classes))
    else:
        train_counts_per_main_class_id = np.zeros(len(classes), dtype=int)

    if y_val.size > 0:
        val_counts_per_main_class_id = np.bincount(y_val.astype(int), minlength=len(classes))
    else:
        val_counts_per_main_class_id = np.zeros(len(classes), dtype=int)

    print("\n--- Train/Validation Split Details (Samples per Subfolder) ---")
    # `classes` is the sorted list of main class names from load_data
    # `subfolder_to_indices_map` has keys like "ClassName/SubfolderName"
    # `train_samples_per_subfolder` and `val_samples_per_subfolder` store counts

    for i, main_class_name in enumerate(classes):
        print(f"\n{main_class_name}:")
        
        # Find all subfolders belonging to this main_class_name
        subfolders_in_main_class = sorted([
            sf_key for sf_key in subfolder_to_indices_map 
            if sf_key.startswith(main_class_name + "/")
        ])

        if not subfolders_in_main_class:
            print(f"  No subfolders found for this class.")
        else:
            for subfolder_key in subfolders_in_main_class:
                num_train_sf = train_samples_per_subfolder.get(subfolder_key, 0)
                num_val_sf = val_samples_per_subfolder.get(subfolder_key, 0)
                total_sf = num_train_sf + num_val_sf
                print(f"  Subfolder {subfolder_key.split('/')[-1]}: Total={total_sf} -> Train={num_train_sf}, Val={num_val_sf}")
        
        print(f"  Total Train Samples for {main_class_name}: {train_counts_per_main_class_id[i]}")
        print(f"  Total Val Samples for {main_class_name}  : {val_counts_per_main_class_id[i]}")
    print("\n")

    # Print dataset shapes
    print("--- Dataset Shapes (Samples, Features) ---")
    print(f"  Raw Training Data (X_tr_raw):   {X_tr_raw.shape}, Labels (y_tr): {y_tr.shape}") # y_tr is np.array
    print(f"  Raw Validation Data (X_val_raw): {X_val_raw.shape}, Labels (y_val): {y_val.shape}") # y_val is np.array
    print(f"  Raw Prediction Data (X_test_raw):     {X_test_raw.shape}, Labels (y_test): {y_test.shape}") # y_test is np.array
    print("\n")


    # 7.4 Preprocessing independently
    X_tr = snv(derivative(det(X_tr_raw)))    # İstenen Sıra: det -> derivative -> snv
    X_val= snv(derivative(det(X_val_raw)))   # İstenen Sıra: det -> derivative -> snv
    X_test= snv(derivative(det(X_test_raw))) # İstenen Sıra: det -> derivative -> snv

    # 7.5 DataLoaders
    train_loader=DataLoader(SpectralDataset(X_tr,y_tr),batch_size=batch_size,shuffle=True, num_workers=4, pin_memory=True)
    val_loader  =DataLoader(SpectralDataset(X_val,y_val),batch_size=batch_size,shuffle=False, num_workers=4, pin_memory=True)
    test_loader =DataLoader(SpectralDataset(X_test,y_test),batch_size=batch_size,shuffle=False, num_workers=4, pin_memory=True)

    # 7.6 Model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_CuiFearn_Adapted(X_tr.shape[1],len(classes)).to(device)

    print("\n--- Model Information ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}") # Bu satır zaten vardı, doğru.
    print("Layer output shapes will be printed ONCE during the first forward pass.") # Mesajı güncelledim
    print("-------------------------")
    # Optimizer Adadelta olarak değiştirildi, Adadelta genellikle öğrenme oranı parametresini bu şekilde almaz.
    optimizer = optim.AdamW(model.parameters())
    scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5,verbose=True)
    
    # AMP için GradScaler
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # 7.7 Training
    train_losses,val_losses,train_accs,val_accs=[],[],[],[]
    best_val_loss, best_epoch, pat = float('inf'), 0, 0
    best_train_loss, best_train_acc, best_val_acc = float('inf'), 0.0, 0.0

    # Hedef dizini oluştur (eğer yoksa)
    os.makedirs(target_save_dir, exist_ok=True)
    print(f"Outputs will be saved to: {target_save_dir}")
    best_model_path = os.path.join(target_save_dir, 'best_model.pth') # best_model.pth yolunu güncelle

    for epoch in range(1,epochs+1):
        model.train()
        rl,rc=0,0
        for Xb,yb in train_loader:
            Xb,yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # AMP ile forward pass
            with autocast(enabled=torch.cuda.is_available()):
                out=model(Xb)
                loss=criterion(out,yb)
            
            # AMP ile backward pass ve optimizer adımı
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            rl+=loss.item()*Xb.size(0);rc+=(out.argmax(1)==yb).sum().item()
        tr_loss, tr_acc = rl/len(train_loader.dataset), rc/len(train_loader.dataset)
        model.eval();vl,vc=0,0
        with torch.no_grad():
            for Xb,yb in val_loader:
                Xb,yb = Xb.to(device), yb.to(device)
                with autocast(enabled=torch.cuda.is_available()): # İsteğe bağlı: Doğrulama için de AMP
                    out=model(Xb)
                    loss=criterion(out,yb)
                vl+=loss.item()*Xb.size(0);vc+=(out.argmax(1)==yb).sum().item()
        val_loss,val_acc=vl/len(val_loader.dataset),vc/len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss = tr_loss # En iyi train loss'u sakla
            best_train_acc = tr_acc   # En iyi train acc'yi sakla
            best_val_acc = val_acc     # En iyi val acc'yi sakla
            pat = 0
            torch.save(model.state_dict(), best_model_path)
        else: pat+=1
        if pat>=early_stop_patience: print(f'Early stopping @ {epoch}');break
        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc); val_accs.append(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs} — LR: {current_lr:.6f} — Train L={tr_loss:.4f}, A={tr_acc:.4f} | Val L={val_loss:.4f}, A={val_acc:.4f}")

    # 7.8 Metrics plots (Combined)
    print("\n--- Plotting Training Metrics ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Create a figure with 2 subplots side-by-side

    axes[0].plot(train_losses, label='Train Loss'); axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss per Epoch')
    axes[0].legend()

    axes[1].plot(train_accs, label='Train Acc'); axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy per Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(target_save_dir, f"training_metrics_epoch{best_epoch if best_epoch > 0 else epochs}.png")) # Grafik kaydetme yolunu güncelle
    plt.show() # Display the combined figure

    if best_epoch > 0:
        print(f"\n--- Best Model Achieved ---")
        print(f"  Epoch: {best_epoch}")
        print(f"  Train Loss: {best_train_loss:.4f}")
        print(f"  Train Acc:  {best_train_acc:.4f}")
        print(f"  Val Loss:   {best_val_loss:.4f}")
        print(f"  Val Acc:    {best_val_acc:.4f}")

    # 7.9 Testing & report
    if best_epoch > 0: # Use best model for prediction
        print(f"\nLoading best model from {best_model_path} (Epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}) for testing...")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("\nNo best model saved during training. Using the model from the last epoch for testing (if any).")
    model.eval()
    all_t, all_p, mis = [], [], []
    ti=0
    with torch.no_grad():
        for Xb,yb in test_loader:
            Xb=Xb.to(device)
            logits=model(Xb);probs=F.softmax(logits,dim=1);preds=probs.argmax(1)
            for i in range(len(yb)):
                t, p = yb[i].item(), preds[i].item();pr=probs[i,p].item()
                # test_paths[ti] is (class_name, subfolder_name, file_name)
                cls_true,cls_pred=classes[t],classes[p]
                pred_file_info = f"{cls_true}/{test_paths[ti][1]}/{test_paths[ti][2]}"
                all_t.append(t);all_p.append(p)
                if t!=p:
                    inds=[j for j,lbl in enumerate(y_tr) if lbl==p]
                    d=[np.linalg.norm(X_tr[j]-X_test[ti]) for j in inds] # Compare preprocessed data
                    bj=inds[int(np.argmin(d))]
                    train_file_info = f"{cls_pred}/{paths_tr[bj][1]}/{paths_tr[bj][2]}" # cls_pred is the class it was mistaken for
                    mis.append((pred_file_info, train_file_info, pr))
                ti+=1
    acc=accuracy_score(all_t,all_p);print(f"Prediction Acc: {acc*100:.2f}%")
    cm=confusion_matrix(all_t,all_p)
    plt.figure(figsize=(8,6));plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    cm_title = f'Confusion Matrix (Epoch {best_epoch if best_epoch > 0 else "Last"})'
    plt.title(cm_title);plt.colorbar()
    tks=np.arange(len(classes));plt.xticks(tks,classes,rotation=45);plt.yticks(tks,classes)
    thresh=cm.max()/2
    for i,j in np.ndindex(cm.shape):plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>thresh else 'black')
    plt.ylabel('True Class');plt.xlabel('Predicted Class');plt.tight_layout();plt.show()
    if mis:
        # --- Yeni Eklenen Karışıklık Özeti ---
        print("\n--- Karışan Ana/Alt Klasör Özeti (Tahmin vs Eğitim) ---")
        misclassification_counts = {} # Key: "GercekAna/GercekAlt", Value: {"KaristigiAna/KaristigiAlt": sayi}

        for pred_file_info_str, train_file_info_str, prob in mis:
            # pred_file_info_str: "GercekSinif/AltKlasorTahmin/DosyaAdi_a.csv"
            parts_pred = pred_file_info_str.split('/')
            actual_main_sub_pred = f"{parts_pred[0]}/{parts_pred[1]}" # Gerçek Ana/Alt

            # train_file_info_str: "KaristigiSinif/AltKlasorEgitim/DosyaAdi_a.csv"
            parts_train = train_file_info_str.split('/')
            confused_main_sub_train = f"{parts_train[0]}/{parts_train[1]}" # Karıştırıldığı Ana/Alt

            if actual_main_sub_pred not in misclassification_counts:
                misclassification_counts[actual_main_sub_pred] = {}
            
            misclassification_counts[actual_main_sub_pred][confused_main_sub_train] = \
                misclassification_counts[actual_main_sub_pred].get(confused_main_sub_train, 0) + 1

        for actual_group, confused_groups in sorted(misclassification_counts.items()):
            total_misclassified_for_actual_group = sum(confused_groups.values())
            print(f"Prediction'dan {actual_group} ({total_misclassified_for_actual_group} örnek yanlış sınıflandırıldı):")
            for confused_group, count in sorted(confused_groups.items(), key=lambda item: item[1], reverse=True):
                print(f"  -> Train'deki {confused_group} ile {count} kez karıştı.")
        print("---------------------------------------------------------------------------------")
        # --- Karışıklık Özeti Bitişi ---

        print("\n--- Bireysel Hatalı Sınıflandırma Detayları ve Grafikleri ---")
        for pred_file_info_str, train_file_info_str, prob in mis:
            print(f"Prediction file '{pred_file_info_str}' was misclassified as similar to train file '{train_file_info_str}' with probability {prob*100:.2f}%.")

            # Parse pred_file_info_str: "TrueClass/Subfolder/Filename_a.csv"
            parts_pred = pred_file_info_str.split('/')
            true_class_pred, subfolder_pred, filename_pred = parts_pred[0], parts_pred[1], parts_pred[2]
            pred_plot_label = f"{true_class_pred}/{subfolder_pred}"
            pred_full_path = os.path.join(test_dir, true_class_pred, subfolder_pred, filename_pred)

            # Parse train_file_info_str: "MistakenClass/Subfolder/Filename_a.csv"
            parts_train = train_file_info_str.split('/')
            mistaken_class_train, subfolder_train, filename_train = parts_train[0], parts_train[1], parts_train[2]
            train_plot_label = f"{mistaken_class_train}/{subfolder_train}"
            train_full_path = os.path.join(train_dir, mistaken_class_train, subfolder_train, filename_train)

            # Load raw spectra
            wl_pred_raw, data_pred_raw = load_single_spectrum(pred_full_path, wl_min, wl_max, skiprows)
            wl_train_raw, data_train_raw = load_single_spectrum(train_full_path, wl_min, wl_max, skiprows)

            if data_pred_raw.size == 0 or data_train_raw.size == 0:
                print(f"  Skipping plot for {pred_file_info_str} vs {train_file_info_str} due to loading error or no data in range.")
                continue

            # Preprocess spectra
            data_pred_proc = preprocess_single_spectrum(data_pred_raw.copy()) # Use .copy() to avoid modifying raw
            data_train_proc = preprocess_single_spectrum(data_train_raw.copy())

            # Add concise print statement before plotting
            print(f"\n  Plotting details for misclassified sample:")
            print(f"    - Gerçek (Tahmin Dosyası): {pred_file_info_str}")
            print(f"    - Karıştırıldığı (Eğitim Dosyası): {train_file_info_str}")
            # Plotting
            fig, (ax_raw, ax_proc) = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle(f"Misclassification: Pred '{pred_plot_label}' vs. Train '{train_plot_label}' (Prob: {prob*100:.2f}%)", fontsize=14)

            # Raw Spectra Plot
            ax_raw_twin = ax_raw.twinx()
            ax_raw.plot(wl_pred_raw, data_pred_raw, color='blue', linestyle='-', label=f"Pred: {pred_plot_label}")
            ax_raw_twin.plot(wl_train_raw, data_train_raw, color='red', linestyle='--', label=f"Train: {train_plot_label}")
            ax_raw.set_xlabel("Wavelength (nm)"); ax_raw.set_ylabel("Absorbance (Pred)", color='blue'); ax_raw_twin.set_ylabel("Absorbance (Train)", color='red')
            ax_raw.tick_params(axis='y', labelcolor='blue'); ax_raw_twin.tick_params(axis='y', labelcolor='red')
            lines, labels = ax_raw.get_legend_handles_labels(); lines2, labels2 = ax_raw_twin.get_legend_handles_labels()
            ax_raw.legend(lines + lines2, labels + labels2, loc='upper right'); ax_raw.set_title("Raw Spectra"); ax_raw.grid(True)

            # Processed Spectra Plot
            ax_proc_twin = ax_proc.twinx()
            ax_proc.plot(wl_pred_raw, data_pred_proc, color='green', linestyle='-', label=f"Pred: {pred_plot_label}") # Use original wavelengths for x-axis
            ax_proc_twin.plot(wl_train_raw, data_train_proc, color='purple', linestyle='--', label=f"Train: {train_plot_label}")
            ax_proc.set_xlabel("Wavelength (nm)"); ax_proc.set_ylabel("Processed Value (Pred)", color='green'); ax_proc_twin.set_ylabel("Processed Value (Train)", color='purple')
            ax_proc.tick_params(axis='y', labelcolor='green'); ax_proc_twin.tick_params(axis='y', labelcolor='purple')
            lines_p, labels_p = ax_proc.get_legend_handles_labels(); lines2_p, labels2_p = ax_proc_twin.get_legend_handles_labels()
            ax_proc.legend(lines_p + lines2_p, labels_p + labels2_p, loc='upper right'); ax_proc.set_title("Processed Spectra"); ax_proc.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust for suptitle
            plt.show()
    else: print("No misclassifications on the test set with the best model.")

    # Rename the best model file to include epoch and loss
    if os.path.exists(best_model_path) and best_epoch > 0:
        final_model_filename = f"best_model_epoch{best_epoch}_loss{best_val_loss:.4f}.pth"
        final_model_full_path = os.path.join(target_save_dir, final_model_filename) # Nihai model yolunu güncelle
        os.rename(best_model_path, final_model_full_path)
        print(f"Best model saved as {final_model_full_path}")


if __name__=='__main__': main()

