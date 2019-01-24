#coding=utf-8
"""
数据加载及预处理
基本原理：使用dataset封装数据集，再使用dataloader实现数据的并行加载
分类：训练集和测试集，从训练集取出一小部分作为验证集
"""
import os
import torch as t
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):

	def __init__(self,root,transforms=None,train=True,test=False):
		'''
		获取所有图片的地址，并根据训练，测试，验证三类对数据进行划分
		'''
		self.test = test
		imgs = [os.path.join(root,img) for img in os.listdir(root)] #获取root路径下所有图片的地址

		#print imgs

		#test1:data/test1/1.jpg
		#train:data/train/cat.1.jpg
		if self.test:  #根据不同的分类对图片按序号排序
			imgs = sorted(imgs,key=lambda x: int(x.split('.')[-2].split('/')[-1]))
		else:
			imgs = sorted(imgs,key=lambda x: int(x.split('.')[-2]))

		#print imgs

		imgs_num = len(imgs) #图片数量，数据集的规模

		#print imgs_num
		
		"""进行数据集的划分"""
		if self.test:  
			self.imgs = imgs   #测试集
		elif train:
			self.imgs = imgs[:int(0.8*imgs_num)]  #训练集
		else:
			self.imgs = imgs[int(0.8*imgs_num):]   #验证集
			
		"""数据转换操作，测试验证和训练的数据转换有所区别"""
		if transforms is None:
			normalize = T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
			
			#测试集和验证集
			if self.test or not train:
				self.transforms = T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),normalize])
			else:  #训练集
				self.transforms = T.Compose([T.Resize(256),T.RandomResizedCrop(224),T.RandomHorizontalFlip(),T.ToTensor(),normalize])
				
		
	def __getitem__(self,index):
		"""
		返回一张图片的数据
		如果是测试集，没有图片对应的类别，如1000.jpg返回1000,如果是训练集和验证集，则对应的是dog返回1，猫则返回0
		"""
		img_path = self.imgs[index]
		
		if self.test: #是测试集
			label = int(self.imgs[index].split('.')[-2].split('/')[-1])
		else:
			label = 1 if 'dog' in img_path.split('/')[-1] else 0
			
		data = Image.open(img_path)
		data = self.transforms(data)
		
		return data,label
	
	def __len__(self):
		"""
		返回数据集中所有图片的个数
		"""
		return len(self.imgs)
	

if __name__ == '__main__':
	test = DogCat('/home/gzshan/sgz/dogsVScats/data/test/',test=True)
	print test
	#print test.__getitem__(2)


