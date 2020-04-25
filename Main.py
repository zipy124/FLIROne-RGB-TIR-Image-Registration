import flirimageextractor
import cv2
from TIPA_library.main.thermal_image_processing import optimal_quantization

def extract_and_register(path):
    """
    Extracts the thermal matrix and rgb data from an image taken on a
    FLIROne camera. Also crops and scales the rgb data, to be registered 
    with the thermal matrix.
    
    :param path: Path where image is located
    :type path: str
    :return: Rgb and thermal images in a tuple, in that order
    :rtype: (list[int,int,int],list[int,int])
    """
    flir = flirimageextractor.FlirImageExtractor()

    flir.process_image(path)

    thermal_im = flir.extract_thermal_image()

    # Read the images to be aligned
    rgb_im = cv2.cvtColor(flir.extract_embedded_image(), cv2.COLOR_RGB2BGR)
    rgb_im = rgb_im[185:-90, 120:-86]

    # thermal_im = optimal_quantization(thermal_im)

    scale_percent = 55  # percent of original size
    width = int(rgb_im.shape[1] * scale_percent / 100)
    height = int(rgb_im.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    rgb_im = cv2.resize(rgb_im, dim, interpolation=cv2.INTER_AREA)

    return rgb_im, thermal_im

def stitch(rgb,thermal,grayscale=True):
    """
    Stitches together a thermal and rgb image, optionally in grayscale or not.
    
    :param rgb: rgb image
    :type rgb: list[int,int,int]
    :param thermal: thermal matrix
    :type thermal: list[int,int]
    :param grayscale: If we want to transfer the rgb to grayscale or not
    :type grayscale: bool
    :return: Stiched image, either grayscale single channel or rgb 3 channel
             image. In the case of 3 channels the thermal is duplicated in 
             each channel.
    :rtype: list[int,int,int] or list[int,int]
    """
    
    
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    if grayscale:
        new_im = cv2.hconcat([thermal,rgb_gray])
    else:
        r = cv2.hconcat([thermal,rgb[:,:,0]])
        g = cv2.hconcat([thermal, rgb[:, :, 1]])
        b = cv2.hconcat([thermal, rgb[:, :, 2]])
        new_im = cv2.merge([r, g, b])

    return new_im

if __name__ =="__main__":
    rgb, thermal = extract_and_register("Path to your image here.jpg")
    
    thermal = optimal_quantization(thermal)
    
    stitch(rgb, thermal, False)
