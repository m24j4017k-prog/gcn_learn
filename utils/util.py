import os
import yaml
from pprint import pprint

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def save_arg(arg, processor):
    # save arg
    arg_dict = vars(arg)
    if not os.path.exists(processor.work_dir):
        os.makedirs(processor.work_dir)
    with open('{}/config.yaml'.format(processor.work_dir), 'w') as f:
        pprint(arg_dict)  
        yaml.dump(arg_dict, f)