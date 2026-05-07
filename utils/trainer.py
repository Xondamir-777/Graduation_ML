"""
utils/trainer.py — Цикл обучения, валидация, чекпоинты, логирование
"""

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix, roc_auc_score
)

from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CHECKPOINTS_DIR, LOGS_DIR, NUM_DISEASE_CLASSES
)
from models.rf_climate import RFClimateModel


class Trainer:
    """
    Обучает MultimodalDiseaseModel.
    RF должен быть уже обучён и передан как аргумент.
    """

    def __init__(self, model: nn.Module, rf: RFClimateModel,
                 device: torch.device):
        self.model  = model.to(device)
        self.rf     = rf
        self.device = device

        # ── Loss ──────────────────────────────────────────────────────────
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.reg_loss = nn.MSELoss()

        # ── Optimizer + Scheduler ─────────────────────────────────────────
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "train_f1":   [], "val_f1":   [],
        }
        self.best_val_f1  = 0.0
        self.best_epoch   = 0

    # ── Один шаг: batch → RF proba → forward ─────────────────────────────

    def _rf_proba_tensor(self, climate_batch: torch.Tensor) -> torch.Tensor:
        """Прогоняет climate через RF, возвращает тензор [B, 10]."""
        X_np    = climate_batch.cpu().numpy()
        proba   = self.rf.predict_proba_from_array(X_np)      # [B, 10]
        return torch.tensor(proba, dtype=torch.float32).to(self.device)

    # ── Train epoch ───────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            images  = batch["image"].to(self.device)
            climate = batch["climate"]               # остаётся на CPU для RF
            labels  = batch["label"].to(self.device)
            probs   = batch["prob"].to(self.device)

            rf_proba = self._rf_proba_tensor(climate)

            self.optimizer.zero_grad()
            out = self.model(images, rf_proba)

            loss_cls = self.cls_loss(out["logits"], labels)
            loss_reg = self.reg_loss(out["prob"], probs)
            loss     = loss_cls + 0.1 * loss_reg      # взвешенная сумма

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = out["logits"].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        n     = len(loader.dataset)
        acc   = np.mean(np.array(all_preds) == np.array(all_labels))
        f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"loss": total_loss / n, "acc": acc, "f1": f1}

    # ── Val / Test epoch ──────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs_pred = [], [], []

        for batch in loader:
            images  = batch["image"].to(self.device)
            climate = batch["climate"]
            labels  = batch["label"].to(self.device)
            probs   = batch["prob"].to(self.device)

            rf_proba = self._rf_proba_tensor(climate)
            out  = self.model(images, rf_proba)

            loss = self.cls_loss(out["logits"], labels) + \
                   0.1 * self.reg_loss(out["prob"], probs)

            total_loss += loss.item() * images.size(0)
            preds = out["logits"].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs_pred.extend(out["prob"].cpu().numpy())

        n   = len(loader.dataset)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {
            "loss": total_loss / n, "acc": acc, "f1": f1,
            "preds": all_preds, "labels": all_labels,
            "probs_pred": all_probs_pred,
        }

    # ── Полный цикл обучения ──────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*60}")
        print(f"  Обучение модели  ({NUM_EPOCHS} эпох, device={self.device})")
        print(f"{'='*60}")

        for epoch in range(1, NUM_EPOCHS + 1):
            t0  = time.time()
            tr  = self._train_epoch(train_loader)
            val = self._eval_epoch(val_loader)
            self.scheduler.step()

            # Лог
            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(val["loss"])
            self.history["train_acc"].append(tr["acc"])
            self.history["val_acc"].append(val["acc"])
            self.history["train_f1"].append(tr["f1"])
            self.history["val_f1"].append(val["f1"])

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{NUM_EPOCHS} "
                f"| loss {tr['loss']:.4f}/{val['loss']:.4f} "
                f"| acc {tr['acc']:.4f}/{val['acc']:.4f} "
                f"| F1 {tr['f1']:.4f}/{val['f1']:.4f} "
                f"| lr {lr:.2e} "
                f"| {time.time()-t0:.1f}s"
            )

            # Сохранить лучшую модель
            if val["f1"] > self.best_val_f1:
                self.best_val_f1 = val["f1"]
                self.best_epoch  = epoch
                self._save_checkpoint("best_model.pth")
                print(f"  ✓ Новый лучший Val F1: {self.best_val_f1:.4f}")

        self._save_history()
        print(f"\n  Лучшая эпоха: {self.best_epoch}  Val F1: {self.best_val_f1:.4f}")

    # ── Финальный тест ────────────────────────────────────────────────────

    def evaluate(self, test_loader: DataLoader,
                 class_names: list = None) -> dict:
        # Загрузить лучшую модель
        best_ckpt = CHECKPOINTS_DIR / "best_model.pth"
        if best_ckpt.exists():
            self.model.load_state_dict(
                torch.load(best_ckpt, map_location=self.device)
            )
            print(f"\n[TEST] Загружена лучшая модель: {best_ckpt}")

        res = self._eval_epoch(test_loader)
        print(f"\n[TEST] Loss={res['loss']:.4f}  Acc={res['acc']:.4f}  F1={res['f1']:.4f}")

        if class_names:
            # print("\nClassification Report:")
            # print(classification_report(
            #     res["labels"], res["preds"],
            #     target_names=class_names, zero_division=0
            # ))
            labels_unique = sorted(set(res["labels"]))

            print(classification_report(
                res["labels"],
                res["preds"],
                labels=labels_unique,
                target_names=[class_names[i] for i in labels_unique],
                zero_division=0
            ))

        # Confusion matrix
        cm = confusion_matrix(res["labels"], res["preds"])
        np.save(LOGS_DIR / "confusion_matrix.npy", cm)

        return res

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str):
        torch.save(
            self.model.state_dict(),
            CHECKPOINTS_DIR / name
        )

    def _save_history(self):
        with open(LOGS_DIR / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
