import PIL
import glob
import numpy as np

def adjust_data_range(data, drange_in, drange_out):

    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def make_grid(images, grid_size=None):

	grid_h, grid_w = grid_size
	img_h, img_w = images.shape[1], images.shape[2]
	grid = np.zeros([grid_h*img_h, grid_w*img_w, 3], dtype=images.dtype)
	for idx in range(images.shape[0]):
		x = (idx % grid_w) * img_w
		y = (idx // grid_w) * img_h
		grid[y : y + img_h, x : x + img_w, :] = images[idx]
	return grid

def convert_to_pil_image(image, drange=[0,1]):

    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[2] # grayscale HWC => HW
    image = adjust_data_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, format)

def save_image(image, filename, drange=[0,1], quality=95):

    img = convert_to_pil_image(image, drange)
    if '.jpg' in filename:
        img.save(filename,"JPEG", quality=quality, optimize=True)
    else:
        img.save(filename)

def save_image_grid(images, filename, drange=[0,1], grid_size=None):

    save_image(make_grid(images, grid_size), filename, drange)

def make_training_gif(img_dir):

	image_grids = []
	filenames = sorted(glob.glob('fake_*.jpg'))
	for file in filenames:
		image = PIL.Image.open(file)
		image_grids.append(image)
	image_grids[0].save('vae_training.gif',
                        format='GIF', 
                        append_images=image_grids[1:],
                        save_all=True,
                        duration=1000,loop=0)