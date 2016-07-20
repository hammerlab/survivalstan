import os
from fnmatch import fnmatch
import ntpath
import pkg_resources

## indexes a directory of stan files
## returns as dictionary containing contents of files

def _list_files_in_path(path, pattern = "*.stan"):
    results = []
    for dirname, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                results.append(os.path.join(dirname, name))
    return(results)

def _read_file(filepath, resource = None):
    print(filepath)
    if not(resource):
        with open(filepath, 'r') as myfile:
            data=myfile.read()
    else:
        data = pkg_resources.resource_string(
            resource, filepath)
    return data

def read_files(path, pattern = '*.stan', encoding="utf-8", resource = None):
    files = _list_files_in_path(path = path, pattern=pattern)
    results = {}
    for file in files:
        file_data = {}
        file_data['path'] = file
        file_data['basename'] = ntpath.basename(file)
        file_data['code'] = _read_file(file, resource = resource).decode(encoding)
        results[file_data['basename']] = file_data['code']
    return(results)


