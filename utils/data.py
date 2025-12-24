from typing import Any, Dict, List
import torch


def multitask_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [b["image"] for b in batch]
    labels = [b["label"] for b in batch]
    task_ids = [b["task_id"] for b in batch]
    images = torch.stack(images, dim=0)
    return {"images": images, "labels": labels, "task_ids": task_ids}


def build_task_configs_from_dataframe(df) -> Dict[str, Dict]:
    task_configs: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        task_id = row["task_id"]
        if task_id not in task_configs:
            task_configs[task_id] = {"task_name": row["task_name"], "num_classes": int(row["num_classes"])}
    return task_configs
