import numpy as np
import torch
import cv2
import os

from os.path import expanduser

from skimage import io, transform

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from torchvision import transforms

from PIL import Image

from data_loader import SalObjDataset
from data_loader import RescaleT
from data_loader import ToTensorLab

from model import U2NET

HOME = expanduser("~") + '/'

model_name='u2net'
model_dir = HOME + model_name + '.pth'

output_dir= HOME +'outputs/'

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split(os.sep)[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	filename = d_dir+imidx+'.png'
	imo.save(filename)

	return filename

def remove_bg(images):
	dataset = SalObjDataset(img_name_list=images,lbl_name_list=[],transform=transforms.Compose([RescaleT(320),ToTensorLab(flag=0)]))
	dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=1)

	net = U2NET(3,1)
	
	net.load_state_dict(torch.load(model_dir, map_location='cpu'))

	net.eval()

	outputs = []
	for i, data in enumerate(dataloader):

		inputs = data['image']
		inputs = inputs.type(torch.FloatTensor)
		inputs = Variable(inputs)

		d1,d2,d3,d4,d5,d6,d7= net(inputs)

		pred = d1[:,0,:,:]
		pred = normPRED(pred)

		filename = save_output(images[i],pred,output_dir)
		outputs.append(filename)

		img = cv2.imread(images[i])
		
		mask = cv2.imread(filename,0)
		
		rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
		rgba[:, :, 3] = mask

		cv2.imwrite(filename, rgba) 
		
		del d1,d2,d3,d4,d5,d6,d7

	return outputs

remove_bg([HOME + 'PasswordServer.png'])