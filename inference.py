import os
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, classification_report


LABEL_NAMES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    data_dir: str = "./data/task_data_sci"
    img_root: str = "./data"
    model_name: str = "openai/clip-vit-large-patch14-336"
    save_dir: str = "./checkpoints"

    test_file: str = "task02_test.tsv"
    output_file: str = "submission_mkan_refine.csv"

    img_size: int = 336
    batch_size: int = 32
    num_workers: int = 4
    num_classes: int = 5
    max_text_len: int = 77

    seeds: List[int] = field(default_factory=lambda: [3407, 42, 2024])

    # Fixed ensemble configuration used for final inference
    ensemble_weights: List[float] = field(
        default_factory=lambda: [0.18749672, 0.18847652, 0.62402676]
    )

    # Optional post-hoc logit adjustment (disabled by default to match the manuscript inference setting)
    bias_config: Dict[int, float] = field(
        default_factory=lambda: {0: 0.1, 4: -0.4}
    )

    use_bias_adjustment: bool = Flase


class CrisisDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: CLIPProcessor,
        transform,
        img_root: str,
        max_text_len: int,
        img_size: int,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.transform = transform
        self.img_root = img_root
        self.max_text_len = max_text_len
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros((3, self.img_size, self.img_size))

        text = str(row["text"]) if "text" in row else str(row["tweet_text"])

        enc = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }


class KANLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features, out_features)
        self.scale_base = nn.Parameter(torch.ones(out_features) * 0.1)
        self.scale_spline = nn.Parameter(torch.ones(out_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(F.silu(x))
        spline_out = self.spline(x * torch.sigmoid(x))
        return base_out * self.scale_base + spline_out * self.scale_spline


class KANDualAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.text_pool = nn.Linear(dim, 1)
        self.score = KANLinear(dim, 1)

    def forward(self, text_tokens: torch.Tensor, vision_tokens: torch.Tensor):
        text_weights = torch.softmax(self.text_pool(text_tokens), dim=1)
        text_context = (text_weights * text_tokens).sum(dim=1, keepdim=True)

        inter_vision = text_context * vision_tokens
        vision_weights = torch.softmax(self.score(inter_vision), dim=1)
        vision_feat = (vision_weights * vision_tokens).sum(dim=1)

        vision_context = vision_tokens.mean(dim=1, keepdim=True)
        inter_text = vision_context * text_tokens
        text_weights_refined = torch.softmax(self.score(inter_text), dim=1)
        text_feat = (text_weights_refined * text_tokens).sum(dim=1)

        return vision_feat, text_feat


class MKANRefine(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        dim = self.clip.config.projection_dim

        self.cross_attn = KANDualAttention(dim)
        self.gate = KANLinear(dim * 2, dim)
        self.classifier = nn.Sequential(
            KANLinear(dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            KANLinear(512, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vision_out = self.clip.vision_model(pixel_values)
        text_out = self.clip.text_model(input_ids, attention_mask=attention_mask)

        vision_tokens = self.clip.visual_projection(vision_out.last_hidden_state)
        text_tokens = self.clip.text_projection(text_out.last_hidden_state)

        vision_global = self.clip.visual_projection(vision_out.pooler_output)
        text_global = self.clip.text_projection(text_out.pooler_output)

        vision_enhanced, text_enhanced = self.cross_attn(text_tokens, vision_tokens)

        vision_final = vision_global + vision_enhanced
        text_final = text_global + text_enhanced

        concat_feat = torch.cat([vision_final, text_final], dim=-1)
        gate = torch.sigmoid(self.gate(concat_feat))
        fused = vision_final + gate * (text_final - vision_final)

        return self.classifier(fused)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_checkpoint(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    return clean_state_dict(state_dict)


def build_dataloader(cfg: Config) -> DataLoader:
    test_path = os.path.join(cfg.data_dir, cfg.test_file)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test_df = pd.read_csv(test_path, sep="\t")
    processor = CLIPProcessor.from_pretrained(cfg.model_name)

    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    dataset = CrisisDataset(
        df=test_df,
        processor=processor,
        transform=transform,
        img_root=cfg.img_root,
        max_text_len=cfg.max_text_len,
        img_size=cfg.img_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def run_single_model_inference(
    cfg: Config,
    checkpoint_path: str,
    loader: DataLoader,
    device: torch.device,
):
    model = MKANRefine(cfg.model_name, cfg.num_classes).to(device)
    state_dict = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference: {os.path.basename(checkpoint_path)}", leave=False):
            logits = model(
                batch["input_ids"].to(device),
                batch["pixel_values"].to(device),
                batch["attention_mask"].to(device),
            )
            logits_list.append(logits.cpu().numpy())
            labels_list.extend(batch["label"].numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(logits_list, axis=0), np.array(labels_list)


def apply_ensemble(logits_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    if len(logits_list) != len(weights):
        raise ValueError("The number of logits arrays must match the number of ensemble weights.")
    final_logits = np.zeros_like(logits_list[0], dtype=np.float32)
    for w, logits in zip(weights, logits_list):
        final_logits += w * logits
    return final_logits


def apply_bias_adjustment(
    logits: np.ndarray,
    bias_config: Optional[Dict[int, float]],
) -> np.ndarray:
    if bias_config is None:
        return logits
    adjusted = logits.copy()
    for cls_idx, bias_val in bias_config.items():
        adjusted[:, cls_idx] += bias_val
    return adjusted


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 40)
    print(f"Final Accuracy: {acc:.5f}")
    print("=" * 40)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for MKAN-Refine")
    parser.add_argument("--data_dir", type=str, default="./data/task_data_sci")
    parser.add_argument("--img_root", type=str, default="./data")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="submission_mkan_refine.csv")
    parser.add_argument("--enable_bias_adjustment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        img_root=args.img_root,
        model_name=args.model_name,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_file=args.output_file,
        use_bias_adjustment=args.enable_bias_adjustment,
    )

    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("MKAN-Refine: Final Evaluation (Weighted Ensemble)")
    print("=" * 60)
    print(f"Model backbone: {cfg.model_name}")
    print(f"Checkpoint directory: {cfg.save_dir}")
    print(f"Ensemble seeds: {cfg.seeds}")
    print(f"Bias adjustment enabled: {cfg.use_bias_adjustment}")

    loader = build_dataloader(cfg)

    all_logits = []
    y_true = None

    for seed in cfg.seeds:
        checkpoint_path = os.path.join(cfg.save_dir, f"ema_seed{seed}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logits, labels = run_single_model_inference(cfg, checkpoint_path, loader, device)
        all_logits.append(logits)

        if y_true is None:
            y_true = labels

    final_logits = apply_ensemble(all_logits, cfg.ensemble_weights)

    if cfg.use_bias_adjustment:
        final_logits = apply_bias_adjustment(final_logits, cfg.bias_config)

    y_pred = final_logits.argmax(axis=1)
    evaluate_predictions(y_true, y_pred)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(cfg.output_file, index=False)
    print(f"Predictions saved to {cfg.output_file}")


if __name__ == "__main__":
    main()
