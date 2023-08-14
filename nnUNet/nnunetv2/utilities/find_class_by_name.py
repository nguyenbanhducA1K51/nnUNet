import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    # what is the purpose of this file
    # module is python file, package is a folder of python file
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            # for a module, hasattr work as module contain class with given classname
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    
    return tr

if __name__=="__main__":
    p1="/root/repo/liver-tumor-segmentation/nnUNet/nnunetv2/experiment_planning"
    p2="DatasetFingerprintExtractor"
    p3="nnunetv2.experiment_planning"
    print (recursive_find_python_class(p1,p2,p3))