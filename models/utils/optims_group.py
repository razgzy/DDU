def set_weight_decay(model):
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    else:
        skip = ()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    else:
        skip_keywords = ()
    has_decay = []
    no_decay = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip) or \
                check_keywords_in_name(name, skip_keywords) or getattr(param, '_no_weight_decay', False):
            no_decay.append(param)
            no_decay_names.append(name)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay, 'name': 'has_decay'},
            {'params': no_decay, 'weight_decay': 0., 'name': 'no_decay'}], no_decay_names 

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin