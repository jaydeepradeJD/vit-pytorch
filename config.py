
from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset
#
__C.DATASET                                = edict()
__C.DATASET.GRAYSCALE                      = False
__C.DATASET.AUTOENCODER                    = False
__C.DATASET.BACKGROUND                     = False

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.BATCH_SIZE                        = 32
__C.CONST.N_VIEWS_RENDERING                 = 5 #1         # Dummy property for Pascal 3D
__C.CONST.NUM_WORKER                        = 8        # number of data workers
__C.CONST.REP                               = 'surface_with_inout' # Represnetation type
#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './logs'
__C.DIR.EXPERIMENT_NAME                     = 'experiment1'
__C.DIR.PROJECT_NAME                        = 'Vision_Language_Protein'
__C.DIR.WEIGHTS                             = None
__C.DIR.AE_WEIGHTS                          = None
#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.AFM_ENCODER                     = False
__C.NETWORK.LATENT_DIM						= 4096
__C.NETWORK.GROUP_NORM                      = None
#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_EPOCHS                        = 500
__C.TRAIN.LEARNING_RATE             		= 0.001 # 0.003 # 3e-4
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = 0.5 #0.3 #.5
__C.TRAIN.SAVE_FREQ                         = 50            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False
__C.TRAIN.LOSS                              = 'bce'
__C.TRAIN.L2_PENALTY                        = 0.0
__C.TRAIN.OPTIM								= 'adam'
__C.TRAIN.GPU								= 4
__C.TRAIN.DEBUG								= False