#!/usr/bin/env python3
"""SageMaker Training Job 入口脚本：四模型对比训练。

SageMaker 路径约定：
  输入数据  /opt/ml/input/data/data/
              ├── train/{Early Blight,Healthy,Late Blight}/
              └── val/{Early Blight,Healthy,Late Blight}/
  模型输出  /opt/ml/model/
              ├── best.pt               ← YOLO best（供 deploy.ipynb 直接用）
              ├── resnet50_best.pt
              ├── mobilenet_best.pt
              ├── cnn_best.pt
              ├── comparison.json
              └── comparison.csv
"""
import argparse, glob, json, os, shutil, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ── 参数 ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models',               default='yolo,resnet50,mobilenet,cnn')
    p.add_argument('--yolo-epochs',          type=int, default=300)
    p.add_argument('--resnet-epochs',        type=int, default=25)
    p.add_argument('--mobile-epochs',        type=int, default=100)
    p.add_argument('--cnn-epochs',           type=int, default=100)
    p.add_argument('--img-size',             type=int, default=224)
    p.add_argument('--batch-size',           type=int, default=16)
    p.add_argument('--early-stop-patience',  type=int, default=8)
    # SageMaker 自动注入
    p.add_argument('--model-dir', default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    p.add_argument('--data-dir',  default=os.environ.get('SM_CHANNEL_DATA', '/opt/ml/input/data/data'))
    return p.parse_args()


# ── Mixup ─────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ── EarlyStopping ─────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=8, delta=1e-4):
        self.patience = patience; self.delta = delta
        self.best = 0.0; self.counter = 0; self.best_state = None

    def step(self, val_acc, model):
        if val_acc > self.best + self.delta:
            self.best = val_acc
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── 自定义 CNN ─────────────────────────────────────────────────────────
class PotatoCNN(nn.Module):
    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def __init__(self, num_classes=3):
        super().__init__()
        self.enc = nn.Sequential(
            self._block(3, 32), self._block(32, 64),
            self._block(64, 128), self._block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.enc(x))


# ── 工具函数 ──────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def make_loaders(data_dir, img_size, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_tf)
    print(f'数据集: train={len(train_ds)}, val={len(val_ds)}, classes={train_ds.class_to_idx}', flush=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def train_pytorch(model, train_loader, val_loader, epochs, lr,
                  use_mixup, patience, device, label):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=patience)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if use_mixup:
                x, ya, yb, lam = mixup_data(x, y)
                loss = mixup_loss(criterion, model(x), ya, yb, lam)
            else:
                loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        if ep % 5 == 0 or ep == 1:
            print(f'[{label}] epoch {ep:3d}/{epochs}  val_acc={val_acc:.4f}  {time.time()-t0:.0f}s', flush=True)

        if es.step(val_acc, model):
            print(f'[{label}] 早停 @ epoch {ep}, best={es.best:.4f}', flush=True)
            break

    if es.best_state:
        model.load_state_dict(es.best_state)
    return es.best, time.time() - t0


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    models_to_train = [m.strip() for m in args.models.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.model_dir, exist_ok=True)

    print(f'device={device}', flush=True)
    print(f'data_dir={args.data_dir}', flush=True)
    print(f'model_dir={args.model_dir}', flush=True)
    print(f'models={models_to_train}', flush=True)

    results = {}

    # DataLoader（PyTorch 模型共用）
    if any(m in models_to_train for m in ['resnet50', 'mobilenet', 'cnn']):
        train_loader, val_loader = make_loaders(args.data_dir, args.img_size, args.batch_size)

    # ── YOLO11n-cls ────────────────────────────────────────────────────
    if 'yolo' in models_to_train:
        from ultralytics import YOLO
        print(f'\n=== YOLO11n-cls  epochs={args.yolo_epochs} ===', flush=True)
        t0 = time.time()
        yolo = YOLO('yolo11n-cls.pt')
        yolo_res = yolo.train(
            task='classify', data=args.data_dir,
            epochs=args.yolo_epochs, imgsz=args.img_size, batch=args.batch_size,
            project='/tmp/yolo_runs', name='train',
            verbose=False,
        )
        yolo_sec = time.time() - t0

        # 找 best.pt 并拷到 model_dir
        candidates = glob.glob('/tmp/yolo_runs/*/weights/best.pt')
        best_src = candidates[-1] if candidates else None
        if best_src and os.path.exists(best_src):
            shutil.copy(best_src, os.path.join(args.model_dir, 'best.pt'))
            size_mb = os.path.getsize(os.path.join(args.model_dir, 'best.pt')) / 1e6
        else:
            size_mb = 0
            print('WARNING: YOLO best.pt not found', flush=True)

        yolo_val_acc = float(yolo_res.results_dict.get('metrics/accuracy_top1', 0))
        results['YOLO11n-cls'] = {
            'val_acc':   round(yolo_val_acc, 4),
            'train_min': round(yolo_sec / 60, 1),
            'params_m':  round(sum(p.numel() for p in yolo.model.parameters()) / 1e6, 2),
            'model_mb':  round(size_mb, 1),
            'notes':     f'epochs={args.yolo_epochs}',
        }
        print(f'YOLO  val_acc={yolo_val_acc:.4f}  {yolo_sec/60:.1f}min', flush=True)

    # ── ResNet50 ───────────────────────────────────────────────────────
    if 'resnet50' in models_to_train:
        print(f'\n=== ResNet50  epochs={args.resnet_epochs} ===', flush=True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.in_features, 3)

        for p in resnet.parameters(): p.requires_grad = False
        for p in resnet.fc.parameters(): p.requires_grad = True
        acc_w, sec_w = train_pytorch(resnet, train_loader, val_loader,
                                     min(5, args.resnet_epochs), 1e-3, False,
                                     args.early_stop_patience, device, 'ResNet50-warmup')

        for p in resnet.parameters(): p.requires_grad = True
        acc_f, sec_f = train_pytorch(resnet, train_loader, val_loader,
                                     args.resnet_epochs, 3e-4, True,
                                     args.early_stop_patience, device, 'ResNet50')

        out = os.path.join(args.model_dir, 'resnet50_best.pt')
        torch.save(resnet.state_dict(), out)
        results['ResNet50'] = {
            'val_acc':   round(max(acc_w, acc_f), 4),
            'train_min': round((sec_w + sec_f) / 60, 1),
            'params_m':  round(count_params(resnet), 2),
            'model_mb':  round(os.path.getsize(out) / 1e6, 1),
            'notes':     f'ImageNet pretrained, epochs={args.resnet_epochs}+warmup5',
        }
        print(f'ResNet50  val_acc={max(acc_w, acc_f):.4f}', flush=True)

    # ── MobileNetV3 ────────────────────────────────────────────────────
    if 'mobilenet' in models_to_train:
        print(f'\n=== MobileNetV3  epochs={args.mobile_epochs} ===', flush=True)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        mobilenet.classifier[-1] = nn.Linear(mobilenet.classifier[-1].in_features, 3)
        acc, sec = train_pytorch(mobilenet, train_loader, val_loader,
                                 args.mobile_epochs, 3e-4, True,
                                 args.early_stop_patience, device, 'MobileNetV3')
        out = os.path.join(args.model_dir, 'mobilenet_best.pt')
        torch.save(mobilenet.state_dict(), out)
        results['MobileNetV3'] = {
            'val_acc':   round(acc, 4),
            'train_min': round(sec / 60, 1),
            'params_m':  round(count_params(mobilenet), 2),
            'model_mb':  round(os.path.getsize(out) / 1e6, 1),
            'notes':     f'ImageNet pretrained, epochs={args.mobile_epochs}',
        }
        print(f'MobileNetV3  val_acc={acc:.4f}', flush=True)

    # ── 自定义 CNN ─────────────────────────────────────────────────────
    if 'cnn' in models_to_train:
        print(f'\n=== PotatoCNN  epochs={args.cnn_epochs} ===', flush=True)
        cnn = PotatoCNN(num_classes=3)
        acc, sec = train_pytorch(cnn, train_loader, val_loader,
                                 args.cnn_epochs, 1e-3, True,
                                 args.early_stop_patience, device, 'PotatoCNN')
        out = os.path.join(args.model_dir, 'cnn_best.pt')
        torch.save(cnn.state_dict(), out)
        results['PotatoCNN'] = {
            'val_acc':   round(acc, 4),
            'train_min': round(sec / 60, 1),
            'params_m':  round(count_params(cnn), 2),
            'model_mb':  round(os.path.getsize(out) / 1e6, 1),
            'notes':     f'从零训练, epochs={args.cnn_epochs}',
        }
        print(f'PotatoCNN  val_acc={acc:.4f}', flush=True)

    # ── 保存对比结果 ───────────────────────────────────────────────────
    if results:
        with open(os.path.join(args.model_dir, 'comparison.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        df = (pd.DataFrame(results).T
              .reset_index().rename(columns={'index': 'model'})
              .sort_values('val_acc', ascending=False))
        df['val_acc_pct'] = df['val_acc'].apply(lambda x: f'{float(x)*100:.2f}%')
        df.to_csv(os.path.join(args.model_dir, 'comparison.csv'), index=False)

        print('\n=== 四模型对比 ===', flush=True)
        print(df[['model', 'val_acc_pct', 'train_min', 'params_m', 'model_mb']].to_string(index=False), flush=True)

    print('\n训练完成', flush=True)


if __name__ == '__main__':
    main()
