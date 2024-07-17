# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
def convert2Greyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):
    gray_array=createInitializedGreyscalePixelArray(image_width, image_height,0)
    # Convert to grayscale
    for i in range(image_height):
        for j in range(image_width):
            gray_array[i][j]=px_array_r[i][j]*0.299+px_array_g[i][j]*0.587+px_array_b[i][j]*0.114
    # Stretch
    min_val=min([min(row) for row in gray_array])
    max_val=max([max(row) for row in gray_array])
    tmp=max_val-min_val
    for i in range(image_height):
        for j in range(image_width):
            gray_array[i][j]=math.floor(255*(gray_array[i][j]-min_val)/tmp)
    return gray_array

def getGradient(image_width, image_height, gray_image):
    sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    
    filtered_x = createInitializedGreyscalePixelArray(image_width, image_height,0)
    filtered_y = createInitializedGreyscalePixelArray(image_width, image_height,0)
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            gx = 0
            gy = 0
            for m in range(3):
                for n in range(3):
                    gx += sobel_x[m][n] * gray_image[i + m - 1][j + n - 1]
                    gy += sobel_y[m][n] * gray_image[i + m - 1][j + n - 1]
            filtered_x[i][j] = gx
            filtered_y[i][j] = gy
    differences = [[abs(filtered_x[i][j] - filtered_y[i][j]) // 1 for j in range(image_width)] for i in range(image_height)]
    return differences

def conv(input_image,kernel, kernel_size, sum,image_width, image_height):
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height,0)
    t=kernel_size//2
    for i in range(t, image_height - t):
        for j in range(t, image_width - t):
            weighted_sum = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    weighted_sum += kernel[m][n] * input_image[i + m - t][j + n - t]
            filtered_image[i][j] = weighted_sum // sum # Divide by the sum of kernel values (273)
    return filtered_image

def getDeviation(image_width, image_height, input_image):
    # kernel = [[1, 4, 7, 4, 1],
    #         [4, 16, 26, 16, 4],
    #         [7, 26, 41, 26, 7],
    #         [4, 16, 26, 16, 4],
    #         [1, 4, 7, 4, 1]]
    # sum=273
    kernel = [[1, 2, 4, 2, 1],
            [2, 4, 8, 4, 2],
            [4, 8, 16, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1]]
    sum=100
    # for i in range(4):
    #     input_image=conv(input_image, kernel, 5, sum, image_width, image_height)
    return conv(input_image, kernel, 5, sum, image_width, image_height)

def getGaussian(image_width, image_height, input_image, kernel_size, sigma):
    kernel=createInitializedGreyscalePixelArray(kernel_size,kernel_size,0)
    t=kernel_size//2
    c=1/(2*math.pi*sigma*sigma)
    k=-2*sigma*sigma
    sum=0
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j]=c*math.exp(((i-t)*(i-t)+(j-t)*(j-t))/k)
            sum+=kernel[i][j]
    # kernel=[[kernel[i][j]/sum for j in range(kernel_size)]for i in range(kernel_size)]
    # print(kernel)
    # kernel = [[1, 2, 1],
    #         [2, 4, 2],
    #         [1, 2, 1]]
    # sum=16
    return conv(input_image, kernel, kernel_size, sum, image_width, image_height)
    

def threshold2binary(image_width, image_height, input_image, threshold):
    binary_image = [[255 if input_image[i][j]>=threshold else 0 for j in range(image_width)] for i in range(image_height)]
    return binary_image

def erosion(image_width, image_height, input_image, kernel_size):
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height,0)
    t=kernel_size//2
    for i in range(t, image_height - t):
        for j in range(t, image_width - t):
            flag=True
            for x in range(kernel_size):
                for y in range(kernel_size):
                    if input_image[i-x+t][j-y+t]!=255:
                        flag=False
                        break           
            if flag:
                filtered_image[i][j] = 255
    return filtered_image

def dilation(image_width, image_height, input_image, kernel_size):
    filtered_image = createInitializedGreyscalePixelArray(image_width, image_height,0)
    t=kernel_size//2
    for i in range(t, image_height - t):
        for j in range(t, image_width - t):
            flag=False
            for x in range(kernel_size):
                for y in range(kernel_size):
                    if input_image[i-x+t][j-y+t]==255:
                        flag=True
                        break           
            if flag:
                filtered_image[i][j] = 255
    return filtered_image

def getConnectedComponent(image_width, image_height, input_image):
    visited = createInitializedGreyscalePixelArray(image_width, image_height,False)
    components = []  

    # def dfs(row, col, component):
    #     if row < 0 or row >= image_height or col < 0 or col >= image_width:
    #         return
    #     if visited[row][col] or input_image[row][col] == 0:
    #         return
    #     visited[row][col] = True
    #     component.append((row, col))
    #     dfs(row - 1, col, component)
    #     dfs(row + 1, col, component)
    #     dfs(row, col - 1, component)
    #     dfs(row, col + 1, component)

    def bfs(row, col):
        component = []
        queue = [(row, col)]
        visited[row][col] = True

        while queue:
            curr_row, curr_col = queue.pop(0)
            component.append((curr_row, curr_col))

            neighbors = [(curr_row - 1, curr_col), (curr_row + 1, curr_col), (curr_row, curr_col - 1), (curr_row, curr_col + 1)]

            for neighbor_row, neighbor_col in neighbors:
                if 0 <= neighbor_row < image_height and 0 <= neighbor_col < image_width and not visited[neighbor_row][neighbor_col] and input_image[neighbor_row][neighbor_col] == 255:
                    queue.append((neighbor_row, neighbor_col))
                    visited[neighbor_row][neighbor_col] = True

        return component

    for i in range(image_height):
        for j in range(image_width):
            if not visited[i][j] and input_image[i][j] == 255:
                component = bfs(i, j)
                if component:
                    components.append(component)

    largest_component = None
    max_area = 0
    max_size = 0

    for component in components:
        min_x = min(component, key=lambda p: p[1])[1]
        max_x = max(component, key=lambda p: p[1])[1]
        min_y = min(component, key=lambda p: p[0])[0]
        max_y = max(component, key=lambda p: p[0])[0]
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        aspect_ratio = max(width, height) / min(width, height)
        pixel_density = len(component) / area

        if len(component) > max_size and aspect_ratio <= 1.8 :#and pixel_density >= 0.8:
            largest_component = component
            max_size = len(component)
            # max_area = area
    min_x = min(largest_component, key=lambda p: p[1])[1]
    max_x = max(largest_component, key=lambda p: p[1])[1]
    min_y = min(largest_component, key=lambda p: p[0])[0]
    max_y = max(largest_component, key=lambda p: p[0])[0]
    return largest_component,(min_x,max_x,min_y,max_y)







# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Multiple_barcodes"
    # filename = "Multiple_barcodes"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here
    gray_image=convert2Greyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)

    # option 1
    gradient_image=getGradient(image_width, image_height, gray_image)
    # option 2
    deviation_image=getDeviation(image_width, image_height, gradient_image)
    tmp=gradient_image

    for i in range(5):
        deviation_image=getGaussian(image_width, image_height, deviation_image,5,2)
        # deviation_image=getDeviation(image_width, image_height, deviation_image)
        gradient_image=getGaussian(image_width, image_height, gradient_image,5,2)
    
    gradient_image=threshold2binary(image_width, image_height,gradient_image,200)
    deviation_image=threshold2binary(image_width, image_height,deviation_image,130)

    for i in range(2):
        gradient_image=erosion(image_width, image_height, gradient_image, 3)
        deviation_image=erosion(image_width, image_height, deviation_image, 5)
    deviation_image=erosion(image_width, image_height, deviation_image, 5)
    for i in range(2):
        gradient_image=dilation(image_width, image_height, gradient_image, 5)
        deviation_image=dilation(image_width, image_height, deviation_image, 5)

    gradient_component,gradient_area=getConnectedComponent(image_width, image_height, gradient_image)
    deviation_component,deviation_area=getConnectedComponent(image_width, image_height, deviation_image)
    
    
    px_array = px_array_r
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    #px_array = tmp

    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    # center_x = image_width / 2.0
    # center_y = image_height / 2.0
    # bbox_min_x = center_x - image_width / 4.0
    # bbox_max_x = center_x + image_width / 4.0
    # bbox_min_y = center_y - image_height / 4.0
    # bbox_max_y = center_y + image_height / 4.0
    
    # bbox_min_x=gradient_area[0]
    # bbox_max_x=gradient_area[1]
    # bbox_min_y=gradient_area[2]
    # bbox_max_y=gradient_area[3]

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    # rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
    #                  edgecolor='g', facecolor='none')
    # axs1[1, 1].add_patch(rect)
    
    bbox_min_x=gradient_area[0]
    bbox_max_x=gradient_area[1]
    bbox_min_y=gradient_area[2]
    bbox_max_y=gradient_area[3]
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    bbox_min_x=deviation_area[0]
    bbox_max_x=deviation_area[1]
    bbox_min_y=deviation_area[2]
    bbox_max_y=deviation_area[3]
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='b', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()