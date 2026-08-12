import torch


def save_checkpoint(path, model, optimizer=None, extra=None):
    state = {"model": model.state_dict(), "stage": model.stage}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    model.set_stage(state.get("stage", "dense"))
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("extra", {})
