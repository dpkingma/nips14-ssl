import numpy as np
import os
import PIL.Image
import pylab

def save_images(images, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    w = sum(i.size[0] for i in images)
    mh = max(i.size[1] for i in images)
    result = PIL.Image.new("RGBA", (w, mh))
    x = 0
    for i in images:
        result.paste(i, (x, 0))
        x += i.size[0]
    result.save(directory+'/'+filename)

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                         scale=True,
                         output_pixel_vals=True,
                         colorImg=False):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
    X = X * 1.0 # converts ints to floats
    
    if colorImg:
        channelSize = X.shape[1]/3
        X = (X[:,0:channelSize], X[:,channelSize:2*channelSize], X[:,2*channelSize:3*channelSize], None)
    
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        
        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype='uint8' if output_pixel_vals else out_array.dtype
                        ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                xi = X[i]
                if scale:
                    xi = (X[i] - X[i].min()) / (X[i].max() - X[i].min())
                out_array[:, :, i] = tile_raster_images(xi, img_shape, tile_shape, tile_spacing, False, output_pixel_vals)
        
    
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        tmp = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                        this_img = scale_to_unit_interval(tmp)
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                        tile_row * (H+Hs): tile_row * (H + Hs) + H,
                        tile_col * (W+Ws): tile_col * (W + Ws) + W
                        ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array

# Matrix to image
def mat_to_img(w, dim_input, scale=False, colorImg=False, tile_spacing=(1,1), tile_shape=0):
    if tile_shape == 0:
        rowscols = int(w.shape[1]**0.5)
        tile_shape = (rowscols,rowscols)
    imgs = tile_raster_images(X=w.T, img_shape=dim_input, tile_shape=tile_shape, tile_spacing=tile_spacing, scale=scale, colorImg=colorImg)
    return PIL.Image.fromarray(imgs)

# Show filters
def imgshow(plt, w, dim_input, scale=False, colorImg=False, convertImgs=False, tile_spacing=(1,1)):
    if convertImgs:
        channelSize = w.shape[0]/3
        w = tuple([w[channelSize*i:channelSize*(i+1)] for i in range(3)])
    plt.axis('Off')
    pil_image = mat_to_img(w, dim_input, scale, colorImg, tile_spacing)
    plt.imshow(pil_image, cmap=pylab.gray(), origin='upper')
    return pil_image


