import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import json 

class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None,labels_path=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.average_codes = torch.load(opts.class_embedding_path, map_location=torch.device("cpu"))
		self.opts = opts
		self.unseen_label_in_test= opts.unseen_label_in_test

		self.source_root = source_root 

		self.path_to_label = None
		if labels_path is not None:
			with open(labels_path) as f:
				labels = json.load(f)["labels"]
			self.path_to_label = {path: label for path, label in labels}
	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		cate = from_path.split('/')[-1].split('_')[0]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im
		if self.path_to_label is not None:
			if not self.unseen_label_in_test:
				arch_fname = os.path.relpath(from_path,self.source_root)
				label = self.path_to_label[arch_fname]
			elif arch_fname not in self.path_to_label:
				label = 0  # assign category 0 to unseen label
			else:
				label = 0
    #label = label[:, label_list]
			return from_im, to_im, self.average_codes[cate] , int(label)
		return from_im, to_im, self.average_codes[cate]
