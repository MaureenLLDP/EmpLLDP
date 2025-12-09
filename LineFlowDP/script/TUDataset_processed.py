import sys
import os

# 获取当前脚本的绝对路径（如 D:\科研\LineFlowDP-main\script\TUDataset_raw.py）
current_script_path = os.path.abspath(__file__)
# 推导项目根目录路径（LineFlowDP-main/）
project_root = os.path.dirname(os.path.dirname(current_script_path))  # 向上两级目录

# 将项目根目录添加到 Python 模块搜索路径
sys.path.append(project_root)


from script.my_util import *

data_path = './data/'


def tudataset(project, version):
    dataset_train = MYDataset(root=(data_path + project), name=version, use_node_attr=True)
    return dataset_train


if __name__ == '__main__':
    for project in list(all_releases.keys()):
        cur_releases = all_releases[project]
        for release in cur_releases:
            tudataset(project=project, version=release)
            print(release, 'done')
