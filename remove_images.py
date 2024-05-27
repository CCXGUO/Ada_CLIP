import os
import shutil
import re
from pathlib import Path

def remove_images_evenly(directory, num_to_remove):
    # 获取所有图片文件
    all_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])

    total_files = len(all_files)
    if num_to_remove >= total_files:
        raise ValueError("Number of files to remove is greater than or equal to the total number of files")

    # 计算需要保留的图片索引
    num_to_keep = total_files - num_to_remove
    step = total_files / num_to_keep

    indices_to_keep = {int(i * step) for i in range(num_to_keep)}

    # 创建一个临时目录存放保留的图片
    temp_directory = os.path.join(directory, "temp")
    os.makedirs(temp_directory, exist_ok=True)

    for index, filename in enumerate(all_files):
        if index in indices_to_keep:
            shutil.move(os.path.join(directory, filename), os.path.join(temp_directory, filename))

    # 清空原目录并将保留的图片移回去
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    for filename in os.listdir(temp_directory):
        shutil.move(os.path.join(temp_directory, filename), os.path.join(directory, filename))

    os.rmdir(temp_directory)
    print(f"Removed {num_to_remove} images. {len(indices_to_keep)} images remain.")


def rename_frames(directory):
    # 使用正则表达式匹配文件名中的数字部分
    pattern = re.compile(r'frame_(\d+)\.png')

    # 获取所有匹配的文件并排序
    frames = sorted(Path(directory).glob('frame_*.png'), key=lambda x: int(pattern.search(x.name).group(1)))

    # 重命名文件
    for i, frame in enumerate(frames):
        new_name = f'frame_{i}.png'
        new_path = frame.with_name(new_name)
        frame.rename(new_path)
        print(f'Renamed {frame.name} to {new_name}')


# 使用方法
directory = '/home/luno/dev/Ada_CLIP/validation_threshold/Video1'  # 替换为你的文件夹路径
num_to_remove = 68

