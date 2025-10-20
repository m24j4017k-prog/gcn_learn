from utils.util import import_class

def build_model(model_path: str, model_args: dict):
    ModelClass = import_class(model_path)
    model = ModelClass(**model_args).cuda()
    return model, ModelClass