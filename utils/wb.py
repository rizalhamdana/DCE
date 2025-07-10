import socket
import os

def get_root_path():
    host_name = socket.gethostname()
    path = "/user/result_path"
    return host_name, path



def get_result_path():
    HOST_NAME, ROOT_PATH = get_root_path()
    path = os.path.join(ROOT_PATH, "result")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def make_new_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)