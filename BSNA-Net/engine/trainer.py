import torch
from utils.batch import move_to_device, split_batch, model_forward


def train_one_epoch(model, loader, optimizer, objective, task, device, sparse=True, grad_clip=None):
    model.train()
    model.set_stage("sparse" if sparse else "dense")
    totals = {"loss": 0.0, "task": 0.0, "sparse": 0.0, "map": 0.0, "div": 0.0, "budget": 0.0}
    samples = 0
    correct = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        inputs, target = split_batch(batch, task)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model_forward(model, inputs, return_aux=True)
        if sparse:
            losses = objective(logits, target, model, aux["subset"])
        else:
            task_loss = torch.nn.functional.cross_entropy(logits, target)
            zero = task_loss.detach() * 0
            losses = {"loss": task_loss, "task": task_loss, "sparse": zero, "map": zero, "div": zero, "budget": zero}
        losses["loss"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        b = target.size(0)
        samples += b
        correct += (logits.argmax(dim=1) == target).sum().item()
        for key in totals:
            totals[key] += float(losses[key].detach()) * b
    metrics = {k: v / max(samples, 1) for k, v in totals.items()}
    metrics["accuracy"] = correct / max(samples, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, task, device, sparse=True):
    model.eval()
    model.set_stage("sparse" if sparse else "dense")
    samples = 0
    correct = 0
    loss_total = 0.0
    activation_total = 0.0
    activation_count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        inputs, target = split_batch(batch, task)
        logits, aux = model_forward(model, inputs, return_aux=True)
        loss = torch.nn.functional.cross_entropy(logits, target)
        b = target.size(0)
        samples += b
        correct += (logits.argmax(dim=1) == target).sum().item()
        loss_total += float(loss) * b
        for mask in aux["masks"].values():
            activation_total += float(mask.mean())
            activation_count += 1
    return {
        "loss": loss_total / max(samples, 1),
        "accuracy": correct / max(samples, 1),
        "activation_ratio": activation_total / max(activation_count, 1)
    }
