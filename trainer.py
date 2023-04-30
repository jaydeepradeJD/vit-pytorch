import pytorch_lightning as pl
import torch
from vit_pytorch import ViT, MAE

class Model(pl.LightningModule):
	"""docstring for ClassName"""
	def __init__(self, cfg):
		super(Model, self).__init__()
		self.cfg = cfg
		self.vit = ViT(image_size = 224,
					   patch_size = 32,
					   num_classes = 1000,
					   dim = 1024,
					   depth = 6,
					   heads = 8,
					   mlp_dim = 2048)
		
		self.mae = MAE(encoder = self.vit,
					   masking_ratio = 0.75,   # the paper recommended 75% masked patches
					   decoder_dim = 512,      # paper showed good results with just 512
					   decoder_depth = 6       # anywhere from 1 to 8
					   )

		self.loss = torch.nn.BCELoss()

	def forward(self, img):
		loss = self.mae(img)
		return loss

	def training_step(self, batch, batch_idx):
		imgs = batch 
		loss = self.forward(imgs)
		self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		imgs = batch 
		loss = self.forward(imgs)
		self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE
		opt = torch.optim.Adam(list(self.mae.parameters()), lr)
			  
		return opt



