import os


def check_mkdir(dir_name: str):
    """
    确保目录存在，不存在则递归创建。
    Args:
        dir_name: 目标目录路径
    """
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
