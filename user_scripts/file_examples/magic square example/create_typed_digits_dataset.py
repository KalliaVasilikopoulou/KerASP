### Delete fontList cache file before running this file ###

import os

found_matplotlib = False

dr = 'C:\\Users'
sub_drs = os.listdir('C:\\Users')

for sub_dr in sub_drs:
    whole_dr = dr + '\\' + sub_dr
    if os.path.isdir(whole_dr):
        try:
            if '.matplotlib' in os.listdir(whole_dr):
                found_matplotlib = True
                whole_dr += '\\.matplotlib'
                if os.listdir(whole_dr) == []:
                    print('.matplotlib folder is empty')
                else:
                    print('.matplotlib folder is not empty')
                    if os.listdir(whole_dr) == ['fontlist-v330.json']:
                        print('fontlist-v330.json has been found and will be deleted')
                        os.remove(whole_dr+'\\fontlist-v330.json')
                break
        except PermissionError:
            continue
        
if not found_matplotlib:
    print('.matplotlib folder was not found')
    print('please make sure to delete C:\\Users\\{YourUsername}\\.matplotlib\\fontlist-v330.json cache file from your computer before running this file for better results')
    
### Run the file ###
        
import numpy as np
import matplotlib.pyplot as plt

import io
from PIL import Image

import matplotlib.font_manager


img = np.zeros((28,28), dtype='uint8')

def typed_digit_array(digit, fontfamily):
    
    # plot figure
    fig = plt.figure()
    fig.set_size_inches((1,1))
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.text(13, 16, str(digit), horizontalalignment='center', verticalalignment='center', color='w', fontfamily=fontfamily, fontsize=90)

    ax.imshow(img, aspect='equal', cmap='gray')

    # save figure to a numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=28)
    buf.seek(0)

    digit_array = np.array(Image.open(buf).convert("L")) # (28,28)

    buf.close()
    plt.close('all')

    return digit_array


# import all possible fonts

font_list = matplotlib.font_manager.findSystemFonts()
font_names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in font_list]

print('Found', len(font_names), 'different fonts.')

typed_digits_images = []
typed_digits_labels = []


from tqdm import tqdm

iterations = 9*len(font_names)
pbar = tqdm(range(iterations),total = iterations , ncols= 100, desc ='Typed MNIST digits dataset creation', position=0, leave=True)

for digit in range(1,10):
    digit_arrays = []
    for font_name in font_names:
        digit_array = typed_digit_array(digit, font_name)   # (28,28)
        for ar in digit_arrays:
            if (digit_array == ar).all():
                continue
        digit_arrays.append(digit_array)
        typed_digits_images.append(digit_array)
        typed_digits_labels.append(digit)
        pbar.update(1)

pbar.close()

typed_digits_images = np.array(typed_digits_images)
typed_digits_labels = np.array(typed_digits_labels)


print('typed_digits_images dataset shape:', typed_digits_images.shape)
print('typed digits labels dataset shape:', typed_digits_labels.shape)









