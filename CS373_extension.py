from pyzbar.pyzbar import decode

import CS373_barcode_detection as d
import imageIO

def detect(image_width, image_height, gray_image):
    
    # option 1
    gradient_image=d.getGradient(image_width, image_height, gray_image)
    # option 2
    deviation_image=d.getDeviation(image_width, image_height, gradient_image)

    for i in range(5):
        deviation_image=d.getGaussian(image_width, image_height, deviation_image,5,2)
        # deviation_image=getDeviation(image_width, image_height, deviation_image)
        gradient_image=d.getGaussian(image_width, image_height, gradient_image,5,2)
    
    gradient_image=d.threshold2binary(image_width, image_height,gradient_image,200)
    deviation_image=d.threshold2binary(image_width, image_height,deviation_image,130)

    for i in range(2):
        gradient_image=d.erosion(image_width, image_height, gradient_image, 3)
        deviation_image=d.erosion(image_width, image_height, deviation_image, 5)
    deviation_image=d.erosion(image_width, image_height, deviation_image, 5)
    for i in range(2):
        gradient_image=d.dilation(image_width, image_height, gradient_image, 5)
        deviation_image=d.dilation(image_width, image_height, deviation_image, 5)

    gradient_component,gradient_area=d.getConnectedComponent(image_width, image_height, gradient_image)
    deviation_component,deviation_area=d.getConnectedComponent(image_width, image_height, deviation_image)

    min_x=min(gradient_area[0],deviation_area[0])-10
    max_x=max(gradient_area[1],deviation_area[1])+10
    min_y=min(gradient_area[2],deviation_area[2])-10
    max_y=max(gradient_area[3],deviation_area[3])+10
    return min_x,max_x,min_y,max_y

def flatten_array(arr):
    flattened = []
    for row in arr:
        for element in row:
            flattened.append(element)
    return flattened

def crop(input_image,min_x,max_x,min_y,max_y):
    output=[]
    for j in range(min_x,max_x):
        for i in range(min_y,max_y):        
            output.append(input_image[i][j])
    return bytes(output)


#filename = "exp"
filename = "Barcode1"
input_filename = "images/"+filename+".png"
# detect(input_filename)
# image_reader = imageIO.png.Reader(filename=input_filename)
# (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

(image_width, image_height, px_array_r, px_array_g, px_array_b) = d.readRGBImageToSeparatePixelArrays(input_filename)
gray_image=d.convert2Greyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)

output_area=detect(image_width, image_height, gray_image)

output=crop(gray_image,output_area[0],output_area[1],output_area[2],output_area[3])
# output=crop(gray_image,0,image_width-1,0,image_height-1)
# output=bytes(flatten_array(output))

# c=decode((output,image_height-1,image_width-1))
c=decode((output, output_area[3]-output_area[2], output_area[1]-output_area[0]))
print(c)