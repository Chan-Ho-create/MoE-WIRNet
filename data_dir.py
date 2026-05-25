import os
import re


def generate_data_dir_simple(folder_path, output_file="HAZE.txt"):
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    all_files = os.listdir(folder_path)

    image_files = [f for f in all_files
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    image_files.sort(key=lambda x: [int(t) if t.isdigit() else t.lower()
                                    for t in re.split('(\d+)', x)])

    data_dirs = [f"{os.path.basename(folder_path)}/{f}" for f in image_files]

    with open(output_file, 'w') as f:
        for data_dir in data_dirs:
            f.write(data_dir + '\n')

    print(f"生成了 {len(data_dirs)} 个data_dir")
    return data_dirs


import re

generate_data_dir_simple("/root/autodl-tmp/Dataset/test/HAZE/input", output_file="HAZE.txt")
