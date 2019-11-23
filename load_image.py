import os
import sys

import PIL.Image as Image
import numpy as np

def resize_image(image, resize_shape):
    return image.resize(resize_shape)

def convert_image_to_nparray(image):
    image = Image.open(image_path)
    numpy_image = np.asarray(image)
    return numpy_image

def convert_nparray_to_image(numpy_image):
    image = PIL.Image.fromarray(np.uint8(numpy_image))
    return image

def convert_images_to_data(image_dir, resize_shape):
    files = os.listdir(image_dir)
    #data = np.empty((0,resize_shape[0], resize_shape[1], 3), int)
    data = []
    #data = np.array([])

    for file in files:

        image = Image.open(image_dir+'/'+file)
        tmp = 'tmp.jpg'
        if image.format == 'PNG':

            #image.save(tmp)
            #image = Image.open(tmp)
            image = image.convert('RGB')
            #rgb_im.save(tmp)
        #print(image.size)
        resize_image = image.resize(resize_shape)
        #print(resize_image)
        np_image = np.asarray(resize_image)
        #np.append(data, resize_image, axis=0)
        data.append(np_image)
        if os.path.isfile(tmp):
            os.remove(tmp)

    return np.array(data)

if __name__ == "__main__":
    if( len(sys.argv) < 2 ):
        sys.exit()
    image_dir = sys.argv[1]
    resize_shape = (240, 240)
    data_arr = convert_images_to_data(image_dir, resize_shape)
    for data in data_arr:
        print(data.shape)