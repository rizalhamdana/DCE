from methods.dce import DCE


# spromptsltmh 原始的sprompt加上了multihead
# spromptsltat 原始的sprompt加上了加权的multihead
def get_model(model_name, args):
    name = model_name.lower()
    options = {
        "dce": DCE,
    }
    return options[name](args)
