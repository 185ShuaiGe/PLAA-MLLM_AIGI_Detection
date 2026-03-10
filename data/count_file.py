import os
import argparse

def count_files_recursively(folder_path):
    """
    递归统计单个文件夹（含所有子层级）下的txt和图片文件总数
    :param folder_path: 要统计的文件夹路径
    :return: 字典 {'txt': 数量, 'image': 数量, 'total_files': 总文件数}
    """
    stats = {'txt': 0, 'image': 0, 'total_files': 0}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    try:
        for root, dirs, files in os.walk(folder_path):
            stats['total_files'] += len(files)
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext == '.txt':
                    stats['txt'] += 1
                elif file_ext in image_extensions:
                    stats['image'] += 1
    except PermissionError:
        print(f"⚠️  无权限访问 {folder_path}，该文件夹统计结果为0")
        return {'txt': 0, 'image': 0, 'total_files': 0}
    except Exception as e:
        print(f"⚠️  统计 {folder_path} 时出错：{str(e)}，结果为0")
        return {'txt': 0, 'image': 0, 'total_files': 0}

    return stats

def main():
    # 设置命令行参数，默认当前工作目录
    parser = argparse.ArgumentParser(description='按一级子文件夹汇总统计txt和图片文件（含所有子层级）')
    parser.add_argument(
        'dir_path',
        type=str,
        nargs='?',
        default=os.getcwd(),
        help='要统计的根文件夹路径（默认统计当前工作目录）'
    )
    args = parser.parse_args()

    # 处理根目录路径（转为绝对路径，避免混淆）
    root_dir = os.path.abspath(args.dir_path)
    print(f"📌 统计根目录：{root_dir}")
    print("-" * 85)

    # 1. 统计根目录下的直接文件（非子文件夹里的）
    root_files_stats = {'txt': 0, 'image': 0, 'total_files': 0}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    # try:
    #     # 遍历根目录的直接文件
    #     for file in os.listdir(root_dir):
    #         file_path = os.path.join(root_dir, file)
    #         if os.path.isfile(file_path):
    #             root_files_stats['total_files'] += 1
    #             file_ext = os.path.splitext(file)[1].lower()
    #             if file_ext == '.txt':
    #                 root_files_stats['txt'] += 1
    #             elif file_ext in image_extensions:
    #                 root_files_stats['image'] += 1
    # except Exception as e:
    #     print(f"⚠️  统计根目录直接文件时出错：{str(e)}，结果为0")
    #     root_files_stats = {'txt': 0, 'image': 0, 'total_files': 0}

    # 2. 获取根目录下的所有一级子文件夹
    first_level_folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            first_level_folders.append(item_path)

    # 3. 统计每个一级子文件夹（含所有子层级）的文件数
    folder_summary = {}
    total_all = {
        'txt': root_files_stats['txt'],
        'image': root_files_stats['image'],
        'total_files': root_files_stats['total_files']
    }

    # # 输出根目录直接文件（如果有）
    # print("📂 根目录直接文件（非子文件夹内）：")
    # print(f"    文本文件(.txt)：{root_files_stats['txt']} 个")
    # print(f"    图片文件：{root_files_stats['image']} 个")
    # print(f"    直接文件总数：{root_files_stats['total_files']} 个")
    # print("-" * 85)

    # 输出每个一级子文件夹的汇总统计
    print("📂 一级子文件夹汇总统计（含所有子层级）：")
    if not first_level_folders:
        print("    根目录下无一级子文件夹")
    else:
        for folder_path in first_level_folders:
            # 获取一级子文件夹的名称（仅显示最后一级，更简洁）
            folder_name = os.path.basename(folder_path)
            # 递归统计该一级文件夹下所有文件（含所有子层级）
            stats = count_files_recursively(folder_path)
            folder_summary[folder_name] = stats
            # 累加至总计
            total_all['txt'] += stats['txt']
            total_all['image'] += stats['image']
            total_all['total_files'] += stats['total_files']
            # 打印当前一级文件夹的结果
            print(f"\n  一级文件夹：{folder_name}")
            print(f"    文本文件(.txt)：{stats['txt']} 个（含所有子层级）")
            print(f"    图片文件：{stats['image']} 个（含所有子层级）")
            print(f"    该文件夹下所有文件总数：{stats['total_files']} 个")

    # 输出总计
    print("-" * 85)
    print("📊 全局总计：")
    print(f"  所有文本文件(.txt)总数：{total_all['txt']} 个")
    print(f"  所有图片文件总数：{total_all['image']} 个")
    print(f"  扫描的文件总数：{total_all['total_files']} 个")
    print("-" * 85)

if __name__ == '__main__':
    main()