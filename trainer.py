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
		img_features = self.encoder(imgs)
		# batch_size,320,4,4,4 or batch_size,160,4,4,4
		emd_features = self.mlp(seq_emd)
		# batch_size,1280
		emd_features = emd_features.view(-1, 80, 4, 4)
		combined_features = torch.cat([img_features, emd_features], dim=1)
		predicted = self.decoder(combined_features)

		return predicted

	def training_step(self, batch, batch_idx):
		imgs, seq_emd, target = batch 
		
		predicted = self.forward(imgs, seq_emd)
		
		loss = self.loss(predicted, target)
		self.log_dict({'train/loss':loss})
		return loss
	
	def validation_step(self, batch, batch_idx):
		imgs, seq_emd, target = batch 
		
		predicted = self.forward(imgs, seq_emd)
		loss = self.loss(predicted, target)
		self.log_dict({'val/loss':loss})
		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.TRAIN.LEARNING_RATE
		opt = torch.optim.Adam(list(self.encoder.parameters())+
			list(self.decoder.parameters())+
			list(self.mlp.parameters()), lr)
			  
		return opt



