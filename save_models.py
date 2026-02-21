"""
Model Export Script for Bone Age Assessment
============================================
Exports trained XGBoost, Ridge, StandardScaler, and ensemble weights
to .joblib files for production backend use.

Requires the RSNA dataset at the configured path.
Run once before starting the backend server.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import xgboost as xgb
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\gouth\Downloads\RSNA_dataset"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "boneage-training-dataset")
TRAIN_CSV = os.path.join(BASE_DIR, "boneage-training-dataset.csv")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_bone_age_model.pth")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 384
MAX_AGE_SCALE = 240.0
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEX_FEATURE_SCALE = 5.0


# ==========================================
# MODEL ARCHITECTURE (must match training)
# ==========================================
class ResNetBoneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Linear(num_features + 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self.stage_head = nn.Sequential(
            nn.Linear(num_features + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, img, sex):
        features = self.backbone(img)
        combined = torch.cat([features, sex.unsqueeze(1)], dim=1)
        return self.age_head(combined), self.stage_head(combined)


# ==========================================
# DATASET (must match training)
# ==========================================
class BoneAgeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            img_path = img_path.replace(".png", ".jpg")
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        sex = torch.tensor(row['sex'], dtype=torch.float32)
        age = torch.tensor(row['bone_age'], dtype=torch.float32)
        stage = torch.tensor(row['stage'], dtype=torch.long)
        return image, sex, age, stage


def process_dataframe(csv_path, img_folder):
    df = pd.read_csv(csv_path)

    if 'Case ID' in df.columns:
        df.rename(columns={'Case ID': 'id'}, inplace=True)

    if 'male' in df.columns:
        df['sex'] = df['male'].apply(lambda x: 1 if x else 0)
    elif 'Sex' in df.columns:
        df['sex'] = df['Sex'].apply(lambda x: 1 if x == 'M' else 0)

    if 'boneage' in df.columns:
        df.rename(columns={'boneage': 'bone_age'}, inplace=True)

    df['path'] = df['id'].apply(lambda x: os.path.join(img_folder, f"{x}.png"))

    def get_stage(months):
        if months < 144:
            return 0
        elif months < 216:
            return 1
        else:
            return 2

    df['stage'] = df['bone_age'].apply(get_stage)
    return df


def extract_features(loader, model, device):
    """Extract deep features using the ResNet50 backbone."""
    model.eval()
    feats_list, age_list, sex_list = [], [], []

    with torch.no_grad():
        for imgs, sexes, ages, _ in loader:
            imgs = imgs.to(device)
            f = model.backbone(imgs)
            f = f.view(f.size(0), -1)
            feats_list.append(f.cpu().numpy())
            age_list.append(ages.numpy())
            sex_list.append(sexes.numpy())

    X = np.vstack(feats_list)
    y = np.concatenate(age_list)
    s = np.concatenate(sex_list)

    # Add sex as feature, scaled by SEX_FEATURE_SCALE (matches training)
    X_final = np.hstack([X, s.reshape(-1, 1) * SEX_FEATURE_SCALE])
    return X_final, y


def main():
    # Validate paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ PyTorch model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(BASE_DIR):
        print(f"❌ RSNA dataset not found: {BASE_DIR}")
        print("   Please update BASE_DIR in this script to point to the RSNA dataset.")
        sys.exit(1)

    print(f"✅ Device: {DEVICE}")
    print(f"✅ Model:  {MODEL_PATH}")
    print(f"✅ Data:   {BASE_DIR}")

    # 1. Load PyTorch model
    print("\n📦 Loading PyTorch model...")
    model = ResNetBoneModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("   ✅ Model loaded successfully.")

    # 2. Prepare data
    print("\n📊 Processing dataframes...")
    full_train_df = process_dataframe(TRAIN_CSV, TRAIN_IMG_DIR)
    train_df, temp_df = train_test_split(
        full_train_df, test_size=0.3, stratify=full_train_df['sex'], random_state=42
    )
    val_df, _ = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['sex'], random_state=42
    )
    print(f"   Train: {len(train_df)} | Val: {len(val_df)}")

    # Transforms must exactly match training validation transforms
    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        BoneAgeDataset(train_df, val_tfm), batch_size=BATCH_SIZE, shuffle=False
    )
    val_loader = DataLoader(
        BoneAgeDataset(val_df, val_tfm), batch_size=BATCH_SIZE, shuffle=False
    )

    # 3. Extract features
    print("\n🔬 Extracting TRAIN features...")
    X_train, y_train = extract_features(train_loader, model, DEVICE)
    print(f"   Shape: {X_train.shape}")

    print("🔬 Extracting VAL features...")
    X_val, y_val = extract_features(val_loader, model, DEVICE)
    print(f"   Shape: {X_val.shape}")

    # 4. Train XGBoost
    print("\n🌲 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=5,
        n_jobs=-1, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    print("   ✅ XGBoost trained.")

    # 5. Train Ridge with StandardScaler
    print("📐 Training Ridge...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    ridge = Ridge(alpha=10)
    ridge.fit(X_train_s, y_train)
    print("   ✅ Ridge trained.")

    # 6. Generate validation predictions for weight optimization
    print("\n🧪 Optimizing ensemble weights on validation set...")
    xgb_val_pred = xgb_model.predict(X_val)
    ridge_val_pred = ridge.predict(scaler.transform(X_val))

    dl_val_pred = []
    model.eval()
    with torch.no_grad():
        for imgs, sexes, _, _ in val_loader:
            imgs, sexes = imgs.to(DEVICE), sexes.to(DEVICE)
            p_age, _ = model(imgs, sexes)
            dl_val_pred.extend((p_age * MAX_AGE_SCALE).cpu().squeeze().numpy())
    dl_val_pred = np.array(dl_val_pred)

    val_matrix = np.column_stack((dl_val_pred, xgb_val_pred, ridge_val_pred))

    def mae_loss(weights):
        final_p = np.dot(val_matrix, weights)
        return mean_absolute_error(y_val, final_p)

    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bnds = ((0, 1), (0, 1), (0, 1))
    result = minimize(mae_loss, [0.33, 0.33, 0.33], method='SLSQP', bounds=bnds, constraints=cons)
    opt_weights = result.x

    print("\n✅ OPTIMAL ENSEMBLE WEIGHTS:")
    print(f"   Deep Learning: {opt_weights[0]:.4f}")
    print(f"   XGBoost:       {opt_weights[1]:.4f}")
    print(f"   Ridge:         {opt_weights[2]:.4f}")
    print(f"   Val MAE:       {result.fun:.4f} months")

    # 7. Save all models
    print("\n💾 Saving model artifacts...")
    xgb_path = os.path.join(OUTPUT_DIR, "xgb_model.joblib")
    ridge_path = os.path.join(OUTPUT_DIR, "ridge_model.joblib")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    weights_path = os.path.join(OUTPUT_DIR, "ensemble_weights.joblib")

    joblib.dump(xgb_model, xgb_path)
    joblib.dump(ridge, ridge_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(opt_weights, weights_path)

    print(f"   ✅ {xgb_path}")
    print(f"   ✅ {ridge_path}")
    print(f"   ✅ {scaler_path}")
    print(f"   ✅ {weights_path}")
    print("\n🎉 All model artifacts exported successfully!")
    print("   You can now start the backend server.")


if __name__ == "__main__":
    main()
