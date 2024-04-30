import os

def clean_tensorboard_logs():
    path = 'tensorboard_logs/'
    for _, dirs, files in os.walk(path):
        for file in files:
            if file == 'projector_config.pbtxt':
                os.remove(path+file)
        for dir_i in dirs:
            if dir_i == 'train':
                dirpath = 'tensorboard_logs/train/'
                for _, _, dirfiles in os.walk(dirpath):
                    for dirfile in dirfiles:
                        if dirfile.startswith('events.out.tfevents') or dirfile.startswith('keras_embedding.ckpt') or dirfile == 'checkpoint':
                            os.remove(dirpath+dirfile)
            elif dir_i == 'validation':
                dirpath = 'tensorboard_logs/validation/'
                for _, _, dirfiles in os.walk(dirpath):
                    for dirfile in dirfiles:
                        if dirfile.startswith('events.out.tfevents'):
                            os.remove(dirpath+dirfile)
