import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

# ---------- 训练  ----------    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.drs_ids = []
        self.drd_ids = []
        self.nrs_ids = []
        # self.nrd_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(f"Degradation types: {self.de_type}")

        # self.de_dict = {'drs': 0, 'drd': 1, 'nrs': 2, 'nrd': 3}
        self.de_dict = {'HAZE': 0, 'RAIN': 1, 'SNOW': 2}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'HAZE' in self.de_type:
            self._init_drs_ids()
        if 'RAIN' in self.de_type:
            self._init_drd_ids()
        if 'SNOW' in self.de_type:
            self._init_nrs_ids()
        # if 'nrd' in self.de_type:
        #     self._init_nrd_ids()

        random.shuffle(self.de_type)

    def _init_drs_ids(self):
        # 添加文件存在性检查
        drs = self.args.data_file_dir + "HAZE/HAZE.txt"
        if not os.path.exists(drs):
            raise FileNotFoundError(f"DRS list file not found: {drs}")

        temp_ids = []
        with open(drs, 'r') as f:
            temp_ids += [os.path.join(self.args.drs_dir, id_.strip()) for id_ in f]

        self.drs_ids = [{"clean_id": x, "de_type": 0} for x in temp_ids]
        self.num_drs = len(self.drs_ids)
        print("Total drs Ids : {}".format(self.num_drs))

    def _init_drd_ids(self):
        drd = self.args.data_file_dir + "RAIN/RAIN.txt"
        if not os.path.exists(drd):
            raise FileNotFoundError(f"DRD list file not found: {drd}")

        temp_ids = []

        with open(drd, 'r') as f:
            temp_ids += [os.path.join(self.args.drd_dir, id_.strip()) for id_ in f]

        self.drd_ids = [{"clean_id":x,"de_type":1} for x in temp_ids]
        self.num_drd = len(self.drd_ids)
        print("Total drd Ids : {}".format(self.num_drd))

    def _init_nrs_ids(self):
        nrs = self.args.data_file_dir + "SNOW/SNOW.txt"
        if not os.path.exists(nrs):
            raise FileNotFoundError(f"NRS list file not found: {nrs}")

        temp_ids = []

        with open(nrs, 'r') as f:
            temp_ids += [os.path.join(self.args.nrs_dir, id_.strip()) for id_ in f]

        self.nrs_ids = [{"clean_id": x, "de_type": 2} for x in temp_ids]
        self.num_nrs = len(self.nrs_ids)
        print("Total nrs Ids : {}".format(self.num_nrs))


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        import os
        dir_path, filename = os.path.split(rainy_name)
        if dir_path.endswith("input"):
            base_dir = os.path.dirname(dir_path)
            gt_name = os.path.join(base_dir, "target", filename)
        else:
            gt_name = rainy_name.replace("/input/", "/target/")
        return gt_name


    def _merge_ids(self):
        self.sample_ids = []
        if "HAZE" in self.de_type:
            self.sample_ids+= self.drs_ids
        if "RAIN" in self.de_type:
            self.sample_ids += self.drd_ids
        if "SNOW" in self.de_type:
            self.sample_ids+= self.nrs_ids
        # if "nrd" in self.de_type:
        #     self.sample_ids += self.nrd_ids
        print(f"Total merged samples: {len(self.sample_ids)}")


    def __getitem__(self, idx):
        original_idx = idx
        for _ in range(len(self.sample_ids)):
            sample = self.sample_ids[idx]
            de_id = sample["de_type"]
            
            try:
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

                degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
                degrad_patch = self.toTensor(degrad_patch)
                clean_patch = self.toTensor(clean_patch)

                return [clean_name, de_id], degrad_patch, clean_patch

            except Exception as e:
                print(f"[Warning] Failed loading sample {idx}: {e}")
                idx = (idx + 1) % len(self.sample_ids)
                if idx == original_idx:
                    break

         # raise RuntimeError("All samples failed to load!")

    def __len__(self):
        return len(self.sample_ids)

class PromptValDataset(Dataset):
    def __init__(self, args, val_type='drs'):
        super().__init__()
        self.args = args
        self.val_type = val_type
        # self.de_dict = {'drs': 0, 'drd': 1, 'nrs': 2, 'nrd': 3}
        self.de_dict = {'HAZE': 0, 'RAIN': 1, 'SNOW': 2}
        self.de_id = self.de_dict[val_type]

        # 每个验证集独立的路径
        self.list_path = os.path.join(args.val_dir, f"{val_type.upper()}/{val_type.upper()}.txt")
        if not os.path.exists(self.list_path):
            raise FileNotFoundError(f"Validation list file not found: {self.list_path}")

        # 读取样本列表
        with open(self.list_path, "r") as f:
            self.sample_ids = [os.path.join(args.val_dir, f"{val_type.upper()}", line.strip()) for line in f]

        print(f"✅ Loaded {len(self.sample_ids)} samples for validation set: {val_type}")

        self.toTensor = ToTensor()

    def _get_gt_name(self, rainy_name):
        dir_path, filename = os.path.split(rainy_name)
        if dir_path.endswith("input"):
            base_dir = os.path.dirname(dir_path)
            gt_name = os.path.join(base_dir, "target", filename)
        else:
            gt_name = rainy_name.replace("/input/", "/target/")
        return gt_name

    def __getitem__(self, idx):
        sample_path = self.sample_ids[idx]
        try:
            degrad_img = crop_img(np.array(Image.open(sample_path).convert('RGB')), base=16)
            clean_name = self._get_gt_name(sample_path)
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            H, W = degrad_img.shape[:2]
            ps = self.args.patch_size
            top = max(0, (H - ps) // 2)
            left = max(0, (W - ps) // 2)
            degrad_patch = degrad_img[top:top+ps, left:left+ps]
            clean_patch = clean_img[top:top+ps, left:left+ps]

            degrad_patch = self.toTensor(degrad_patch)
            clean_patch = self.toTensor(clean_patch)

            return [clean_name, self.de_id], degrad_patch, clean_patch

        except Exception as e:
            print(f"Error loading {sample_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.sample_ids))

    def __len__(self):
        return len(self.sample_ids)    
    
    

#################################################################
class DerainDataset(Dataset):
    def __init__(self, args, addnoise=False, sigma=None):
        super(DerainDataset, self).__init__()
        self.ids = []
        self.args = args
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self._init_input_ids()

        self._check_dataset_initialization()

    def _check_dataset_initialization(self):
        
        if self.length == 0:
            print("warning")
            return

        for i in range(min(5, len(self.ids))):
            degraded_path = self.ids[i]
            clean_path = self._get_gt_path(degraded_path)
            print(f"  {i+1}. 雨天图像: {degraded_path}")
            print(f"     存在: {os.path.exists(degraded_path)}")
            print(f"     无雨图像: {clean_path}")
            print(f"     存在: {os.path.exists(clean_path)}")
            
            if os.path.exists(degraded_path):
                try:
                    with Image.open(degraded_path) as img:
                        print(f"     尺寸: {img.size}, 模式: {img.mode}")
                except Exception as e:
                    print(f"     加载失败: {e}")
        print("===========================")

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        self.ids = []
        input_dir = os.path.join(self.args.derain_path, 'input/')
        print(f"去雨任务 - 输入目录: {input_dir}")
        print(f"目录存在: {os.path.exists(input_dir)}")
        
        if os.path.exists(input_dir):
            name_list = os.listdir(input_dir)
            print(f"找到文件数量: {len(name_list)}")
            # 使用os.path.join确保路径正确
            self.ids += [os.path.join(input_dir, id_) for id_ in name_list]
        else:
            print(f"错误: 输入目录不存在 - {input_dir}")
            
        # 检查对应的target目录
        target_dir = os.path.join(self.args.derain_path, 'target/')
        print(f"去雨任务 - 标签目录: {target_dir}")
        print(f"目录存在: {os.path.exists(target_dir)}")
        if os.path.exists(target_dir):
            target_files = os.listdir(target_dir)
            print(f"标签文件数量: {len(target_files)}")

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        """获取对应的无雨图像路径"""
        # 去雨任务：直接将input替换为target
        gt_name = degraded_name.replace("input", "target")
        return gt_name

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)
        
        # 检查文件是否存在
        if not os.path.exists(degraded_path):
            raise FileNotFoundError(f"雨天图像不存在: {degraded_path}")
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f"无雨图像不存在: {clean_path}")

        try:
            # 加载图像
            degraded_img = Image.open(degraded_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')

            # 保存原始尺寸信息
            original_size = degraded_img.size  # (W, H)

            # 转换为numpy数组（保持原始尺寸）
            degraded_array = np.array(degraded_img)
            clean_array = np.array(clean_img)

            # 打印原始尺寸信息
            # print(f"原始图像尺寸 - 雨天: {degraded_array.shape}, 无雨: {clean_array.shape}")

            if self.addnoise:
                degraded_array, _ = self._add_gaussian_noise(degraded_array)

            # 转换为Tensor
            clean_tensor = self.toTensor(clean_array)
            degraded_tensor = self.toTensor(degraded_array)

            degraded_name = os.path.splitext(os.path.basename(degraded_path))[0]

            # 返回时包含原始尺寸信息
            return [degraded_name], degraded_tensor, clean_tensor, original_size
            
        except Exception as e:
            print(f"加载图像时出错:")
            print(f"  雨天图像: {degraded_path}")
            print(f"  无雨图像: {clean_path}")
            print(f"  错误信息: {e}")
            raise

    def __len__(self):
        return self.length








class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    

