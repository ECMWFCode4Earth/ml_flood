import os
import os.path.join as pjoin

def rename_files(path, old, new, str_constraint=None):
    """
    loops through the given directory and replaces the specified part of a filename with the new part
    the str_constrait variable can be used to only include files, which include a specified string
    """
    for name in os.listdir(path):
        if str_constraint:
            if str_constraint in name:
                if old in name:
                    name_new = name.replace(old, new)
                    os.rename(pjoin(path, name), pjoin(path, name_new))
        else:
            if old in name:
                name_new = name.replace(old, new)
                os.rename(pjoin(path, name), pjoin(path, name_new))
