import PIL
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import cv2
from vidaug import augmentors as va
from definition import *
import random
import numpy as np
class S2T_Dataset(Dataset.Dataset):
    def __init__(self, path,tokenizer,config,args,phase,training_refurbish=False):
        '''
        :param path: 配置文件yml里记录的打包好的数据存放的位置 labels.train
        :param tokenizer: 对图片和文本token的token
        :param config: 配置文件yml存放的位置
        :param args: 训练参数们
        :param phase: 控制是不是训练集
        :param training_refurbish: 布尔参数，用于控制是否对目标批次数据（tgt_batch）进行某种形式的扰动或注入噪声
        '''
        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.phase = phase
        self.training_refurbish = training_refurbish


        self.img_path = config['data']['img_path']
        self.kps_path = config['data']['keypoint_path']


        self.max_length = config['data']['max_length']

        self.list = [key for key, value in self.raw_data.items()]  # 取得打包文件的键值对

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        '''
        __getitem__ 是 Python 数据加载类（通常是继承自 PyTorch 的 Dataset 类）中的一个方法。
        这个方法的主要目的是根据给定的索引 index 从数据集中获取一个数据样本。
        这在深度学习模型的训练过程中非常常见，因为模型需要通过迭代器按批次获取数据
        '''
        # 根据索引获取键
        key = self.list[index]
        # 使用键从原始数据字典中获取样本
        sample = self.raw_data[key]
        # 获取样本中的文本目标
        tgt_sample = sample['text']
        # 获取样本中的名称信息
        name_sample = sample['name']
        # 加载样本中的图像数据
        img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']])
        # 这里我要增加关键点，关键点会和image的一一配对
        kp_sample = self.load_kps([self.kps_path+x for x in sample["keypoint_path"]])

        return name_sample,img_sample,kp_sample,tgt_sample

    def load_imgs(self, paths):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        '''
        这段代码会截断到max length 比如500个训练数据选300个      
        '''
        if len(paths) > self.max_length:
            # 由于我的关键点和图像是严格匹配的，所以random sample 是不行的。
            # tmp = sorted(random.sample(range(len(paths)), k=self.max_length))
            # new_paths = []
            # for i in tmp:
            #     new_paths.append(paths[i])
            # paths = new_paths
            paths = paths[:self.max_length]

        imgs = torch.zeros(len(paths), 3, self.args.input_size2, self.args.input_size)
        # 注意！！！！ 因为我们有input_size 和output_size ,而且我取消了resize，所以提前要按照这个大小去处理好数据！
        # 这样我的imgs才能完整存储图像的信息！！ 对于关键点，也要存储成这样，同时要对关键点数据的坐标也提前处理好，保证resize 以后依然能正常匹配到对应关键点
        # 这需要的就是对关键点的数据的处理了。
        batch_image = []
        for i,img_path in enumerate(paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 我怀疑转换成PIL没有意义，因为我不打算做数据增强
            # img = Image.fromarray(img)
            batch_image.append(img)

        # 这里的内容是为了对训练集进行数据增强，data augmentation，我并不打算做数据增强，所以删除
        # if self.phase == "train":
        #     batch_image

        for i, img in enumerate(batch_image):
            # resize 的过程我打算提前在本地做好，而不是交给模型去resize，节省cost ，同时也方便我align 关键点和图像
            # img = img.resize(resize)
            img = data_transforms(img)
            imgs[i] = img
        return imgs
    def load_kps(self, paths):
        '''
        对关键点处理最大的问题 ：
        1. 降低I/O 以便模型读取
        2. 关键点存储是按照图像坐标存储， 图片会提前resize好， 那么关键点的具体位置要按照resize的原理去变换
        3. 关键点不仅仅是单独的某几个关键点，要对周围像素点做扩散，这样才能形成区别的分布， 这个扩散范围的大小也可以设置成可训练的
        4. 还有一个就是对关键点未来* confi 的数值进行训练

        综上，这里加载关键点，我们要加载最简单的resize 以后的关键点矩阵 。 矩阵元素要用confi 去存储
        处理以上问题我目前的一个想法是先绘制一个jpg文件， 和图片数据是对应的，但是区别是只有关键点位置保存原始confi，其他地方
        都是 （0 ，0，0）
        这样读取的就是图像数据，而不是很多很多的json文件。

        按照这个思路的话，我就把load——img的方法再写一遍就行，关键点部分是提前训练好的。
        而且好处是我可以简单的应用图片的resize 到这个上面，因为resize方法一样，所以我不用考虑关键点转换的复杂运算


        :param paths:  存储路径
        :return: 返回关键点的张量， 它和图片张量大小一样，每个元素存储的是对应关键点和附近关键点的confidence
        '''
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        if len(paths) > self.max_length:
            paths = paths[:self.max_length]
        # 设置一个零向量矩阵，准备保存我们的关键点信息
        kps = torch.zeros(len(paths), 3, self.args.input_size2, self.args.input_size)
        # 我本来认为将三维度都设为confi会降低运算效率，但是对比过性能以后发现其实其他维度全加0 并没有明显的降低运算时间
        # 但是我依旧想先用一个维度去保存
        # 但是也可以用三个维度保存去进行训练， 因为这样可能会最大化两个聚类的分布？
        # 具体的还是要实验
        batch_kps = []
        for i, kps_path in enumerate(paths):
            kp = cv2.imread(kps_path)
            kp = cv2.cvtColor(kp, cv2.COLOR_BGR2RGB)

            batch_kps.append(kp)

        for i, kp in enumerate(batch_kps):
            # resize 的过程我打算提前在本地做好，而不是交给模型去resize，节省cost ，同时也方便我align 关键点和图像
            # img = img.resize(resize)
            kp = data_transforms(kp)
            kps[i] = kp

        return kps
    def load_both(self,path1,path2):
        '''
        这个方法我考虑同时加载图像和关键点，但是关键点本身是要去训练的，所以加载它好像没啥意义？

        :param path1:
        :param path2:
        :return:
        '''
        return

    def collate_fn(self, batch):
        tgt_batch, img_tmp, src_length_batch, name_batch, kp_tmp = [], [], [], [], []

        for name_sample, img_sample, kp_sample, tgt_sample in batch:
            name_batch.append(name_sample)
            img_tmp.append(img_sample)
            kp_tmp.append(kp_sample)
            tgt_batch.append(tgt_sample)

        # 计算最大长度
        max_len = max([vid.size(0) for vid in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad

        # 填充图像和关键点
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(max_len - vid.size(0) - left_pad, -1, -1, -1),
            ), dim=0) for vid in img_tmp]

        padded_kp = [torch.cat(
            (
                kp[0][None].expand(left_pad, -1, -1, -1),
                kp,
                kp[-1][None].expand(max_len - kp.size(0) - left_pad, -1, -1, -1),
            ), dim=0) for kp in kp_tmp]

        video_length = torch.LongTensor([padded_video[i].size(0) for i in range(len(padded_video))])

        img_tmp = [padded_video[i][:video_length[i], :, :, :] for i in range(len(padded_video))]
        kp_tmp = [padded_kp[i][:video_length[i], :, :, :] for i in range(len(padded_kp))]

        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))

        src_length_batch = torch.tensor(src_length_batch)

        img_batch = torch.cat(img_tmp, 0)
        kp_batch = torch.cat(kp_tmp, 0)

        new_src_lengths = (((src_length_batch - 5 + 1) / 2) - 5 + 1) / 2
        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()

        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt", padding=True, truncation=True)

        src_input = {
            'img_ids': img_batch,
            'kp_ids': kp_batch,
            'attention_mask': img_padding_mask,
            'name_batch': name_batch,
            'src_length_batch': src_length_batch,
            'new_src_length_batch': new_src_lengths,
        }
        if self.training_refurbish:
            masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type, random_shuffle=self.args.random_shuffle, is_train=(self.phase=='train'))
            with self.tokenizer.as_target_tokenizer():
                masked_tgt_input = self.tokenizer(masked_tgt, return_tensors="pt", padding = True,  truncation=True)
            return src_input, tgt_input, masked_tgt_input
        return src_input, tgt_input
    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
