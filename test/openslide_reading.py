import matplotlib as mpl
import numpy as np
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import openslide


def read_openslide_all(filename, level):
    slide = openslide.open_slide(filename)
    data = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))
    return data


def read_openslide(filename, level, tile_size=None):
    slide = openslide.open_slide(filename)
    metadata = {key.lower(): value for key, value in dict(slide.properties).items()}
    width0, height0 = slide.level_dimensions[0]
    width, height = slide.level_dimensions[level]
    tile_width = int(metadata.get('openslide.level[0].tile-width'))
    tile_height = int(metadata.get('openslide.level[0].tile-height'))
    tile_width0 = tile_width * width0 // width
    tile_height0 = tile_height * height0 // height

    data = np.zeros((height, width, 3), dtype=np.uint8)
    for yi in range(height // tile_height):
        for xi in range(width // tile_width):
            x0 = xi * tile_width0
            y0 = yi * tile_height0
            tile = np.array(slide.read_region((x0, y0), level, (tile_width, tile_height)).convert('RGB'))
            x = xi * tile_width
            y = yi * tile_height
            data[y:y+tile.shape[1], x:x+tile.shape[0], :] = tile

    return data

if __name__ == "__main__":
    filename = 'D:/slides/3DHistech/sample4.mrxs'
    level = 5
    #data = read_openslide_all(filename, level=level)
    data = read_openslide(filename, level=level)
    plt.imshow(data)
    plt.show()

