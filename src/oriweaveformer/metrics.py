import torch


@torch.no_grad()
def pooled_metrics(model, loader, device, amp_dtype):
    model.eval()
    tp = fp = fn = 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            prediction = torch.sigmoid(model(image)["logits"]) >= 0.5
        truth = target > 0.5
        tp += int((prediction & truth).sum())
        fp += int((prediction & ~truth).sum())
        fn += int((~prediction & truth).sum())
    eps = 1e-7
    precision, recall = tp / (tp + fp + eps), tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1,
            "iou": tp / (tp + fp + fn + eps)}
