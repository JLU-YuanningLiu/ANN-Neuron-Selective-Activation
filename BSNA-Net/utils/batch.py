import torch


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


def split_batch(batch, task):
    if task == "vision":
        if isinstance(batch, dict):
            x = batch.get("image", batch.get("pixel_values"))
            y = batch.get("label", batch.get("labels"))
            return {"x": x}, y
        x, y = batch[0], batch[1]
        return {"x": x}, y
    if isinstance(batch, dict):
        y = batch.get("label", batch.get("labels"))
        inputs = {k: v for k, v in batch.items() if k not in {"label", "labels"}}
        return inputs, y
    inputs, y = batch[0], batch[1]
    if isinstance(inputs, dict):
        return inputs, y
    return {"input_ids": inputs}, y


def model_forward(model, inputs, return_aux=False):
    if "x" in inputs:
        return model(inputs["x"], return_aux=return_aux)
    return model(return_aux=return_aux, **inputs)


def model_encode(model, inputs):
    if "x" in inputs:
        return model.encode_input(inputs["x"])
    return model.encode_input(**{k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask"}})
