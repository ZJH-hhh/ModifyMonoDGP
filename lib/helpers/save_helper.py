import os
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state, 'best_result': best_result, 'best_epoch': best_epoch}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        try:
            # PyTorch ≥ 2.6，需要明确 weights_only=False
            checkpoint = torch.load(filename, map_location, weights_only=False)
        except TypeError:
            # PyTorch < 2.6，不支持 weights_only 参数
            checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            # Load state dict with strict=False to handle architecture mismatches
            try:
                model.load_state_dict(checkpoint['model_state'], strict=True)
                logger.info("==> Loaded all weights successfully")
            except RuntimeError as e:
                logger.warning(f"==> Strict loading failed: {str(e)[:200]}...")
                logger.info("==> Attempting flexible loading...")
                
                # Try flexible loading
                model_dict = model.state_dict()
                checkpoint_dict = checkpoint['model_state']
                
                # Filter out mismatched parameters
                matched_dict = {}
                mismatched_keys = []
                missing_keys = []
                
                for k, v in checkpoint_dict.items():
                    if k in model_dict:
                        if v.shape == model_dict[k].shape:
                            matched_dict[k] = v
                        else:
                            mismatched_keys.append(f"{k}: checkpoint{v.shape} vs model{model_dict[k].shape}")
                    else:
                        missing_keys.append(k)
                
                # Check for missing keys in model
                for k in model_dict.keys():
                    if k not in checkpoint_dict:
                        missing_keys.append(f"Model has {k} but checkpoint doesn't")
                
                # Load matched parameters
                model_dict.update(matched_dict)
                model.load_state_dict(model_dict)
                
                logger.info(f"==> Loaded {len(matched_dict)} matching parameters")
                logger.info(f"==> Skipped {len(mismatched_keys)} mismatched parameters")
                if len(mismatched_keys) < 10:  # Only show if not too many
                    for key in mismatched_keys[:5]:
                        logger.info(f"    Mismatched: {key}")
                logger.info(f"==> {len(missing_keys)} parameters not found in checkpoint")
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch