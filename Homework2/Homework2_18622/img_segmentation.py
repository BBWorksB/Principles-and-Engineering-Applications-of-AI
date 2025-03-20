from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

img_path = 'problem1_cat.png' # get this photo from Canvas. file name (or path if image isn't in the same directory)
img = Image.open(img_path) # define img using PIL
img = ImageOps.grayscale(img) # make the image grayscale 
# img.show() # view image
img = np.array(img).astype(np.float32) # convert img to ndarray 
binary_img = (img < 128).astype(int) # binarize the image 
print(img.shape)
# h, w = img.shape # get height and width of the im

# Plot the grayscale image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')

# Plot the binary image
plt.subplot(1, 2, 2)
plt.imshow(binary_img, cmap='gray')
plt.axis('off')
plt.show()


# Define pixel function
def pixel(x, y, w):
    return x * w + y


# Convert the image to graph
def image_to_graph(image):
    height, width = image.shape
    num_pixels = height * width
    capacity = np.zeros((num_pixels, num_pixels))


    for i in range(height):
        for j in range(width):
            index = pixel(i, j, width)


            # check neighbors right and down
            if j < width - 1:
                right_index = pixel(i, j + 1, width)
                weight = 1 / (1 + abs(img[i, j] - img[i, j + 1]))
                capacity[index][right_index] = weight
                capacity[right_index][index] = weight


            if i < height - 1:
                down_index = pixel(i + 1, j, width)
                weight = 1 / (1 + abs(img[i, j] - img[i + 1, j]))
                capacity[index][down_index] = weight
                capacity[down_index][index] = weight

    return capacity


# Impliment BFS
def bfs(capacity, source, valid_nodes):
    visited = np.zeros(len(capacity), dtype=int)
    queue = deque([source])
    visited[source] = 1


    while queue:
        u = queue.popleft()

        for v in range(len(capacity)):
            if(
                not visited[v] and
                capacity[u][v] > 0 and 
                v in valid_nodes
            ):
                queue.append(v)
                visited[v] = 1


    visited[source] = 1

    return visited


# Implement sink and source
def bfs_sink(sink, source, total_nodes):
    visited = np.ones(total_nodes, dtype=int)
    visited[sink] = 0
    visited[source] = 0
    return visited

# impliment source node
def bfs_source_sink(image):
    height, width = image.shape
    source = pixel(0, 0, width)
    sink = pixel(height // 2, width // 2, width)
    return source, sink


# Execute
if __name__ == '__main__':
    capacity = image_to_graph(binary_img)
    source, sink = bfs_source_sink(binary_img)

    # Foreground in black
    foreground_nodes = set(np.where(binary_img.flatten() == 0)[0])


    # Computre reachability
    source_reachable = bfs(capacity, source, foreground_nodes)
    sink_reachable = bfs_sink(sink, source, len(capacity))

    # Reshape image
    final_reach_source = source_reachable.reshape(binary_img.shape)
    final_reach_sink = sink_reachable.reshape(binary_img.shape)


    # Source node be black
    final_reach_source[0, 0] = 0

    # Display reachable nodes
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Reachable from source')
    plt.imshow(final_reach_source, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Reachable from sink')
    plt.imshow(final_reach_sink, cmap='gray')
    plt.axis('off')

    plt.show()



