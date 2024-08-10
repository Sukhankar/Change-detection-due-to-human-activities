from flask import Flask, render_template, request, url_for, session
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import skimage.morphology
import uuid  
import time 

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' 

UPLOAD_FOLDER = './static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_images(image1_path, image2_path, output_dir):
    print('[INFO] Reading Images ...')
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Resize Images
    print('[INFO] Resizing Images ...')
    new_size = (image1.shape[0] // 5 * 5, image1.shape[1] // 5 * 5)  # Ensure dimensions are multiples of 5
    image1 = cv2.resize(image1, (new_size[1], new_size[0])).astype(int)
    image2 = cv2.resize(image2, (new_size[1], new_size[0])).astype(int)

    # Difference Image
    print('[INFO] Computing Difference Image ...')
    diff_image = abs(image1 - image2)
    cv2.imwrite(os.path.join(output_dir, 'difference.jpg'), diff_image)
    diff_image = diff_image[:, :, 1]

    # PCA and Feature Vector Space
    print('[INFO] Performing PCA ...')
    pca = PCA()
    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    pca.fit(vector_set)
    EVS = pca.components_

    print('[INFO] Building Feature Vector Space ...')
    FVS = find_FVS(EVS, diff_image, mean_vec, new_size)
    components = 3

    # Clustering
    print('[INFO] Clustering ...')
    least_index, change_map = clustering(FVS, components, new_size)
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    change_map = change_map.astype(np.uint8)
    change_map_path = os.path.join(output_dir, 'ChangeMap.jpg')
    cv2.imwrite(change_map_path, change_map)

    # Morphological Operations
    print('[INFO] Performing Closing ...')
    kernel = skimage.morphology.disk(6)
    CloseMap = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, 'CloseMap.jpg'), CloseMap)

    print('[INFO] Performing Opening ...')
    OpenMap = cv2.morphologyEx(CloseMap, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, 'OpenMap.jpg'), OpenMap)

    return change_map_path

def find_vector_set(diff_image, new_size):
    vector_set = []
    for j in range(0, new_size[0], 5):
        for k in range(0, new_size[1], 5):
            block = diff_image[j:j + 5, k:k + 5]
            if block.shape == (5, 5):  # Ensure block is of shape 5x5
                feature = block.ravel()
                vector_set.append(feature)
    
    vector_set = np.array(vector_set)
    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new_size):
    feature_vector_set = []
    for i in range(2, new_size[0] - 2):
        for j in range(2, new_size[1] - 2):
            block = diff_image[i-2:i+3, j-2:j+3]
            if block.shape == (5, 5):  # Ensure block is of shape 5x5
                feature = block.flatten()
                feature_vector_set.append(feature)

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    return FVS

def clustering(FVS, components, new_size):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)
    least_index = min(count, key=count.get)
    change_map = np.reshape(output, (new_size[0] - 4, new_size[1] - 4))
    return least_index, change_map

def cleanup_images():
    """Delete old images from the upload folder."""
    now = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > 3600:  # Keep files for 1 hour
                os.remove(file_path)

@app.before_request
def before_request():
    """Run cleanup before handling each request."""
    cleanup_images()

@app.route('/', methods=['GET', 'POST'])
def index():
    image1 = None
    image2 = None
    filename = None
    
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            return 'No file part'
        
        image1_file = request.files['image1']
        image2_file = request.files['image2']

        if image1_file.filename == '' or image2_file.filename == '':
            return 'No selected file'
        
        if image1_file and image2_file:
            # Generate unique filenames
            image1_filename = str(uuid.uuid4()) + "_" + image1_file.filename
            image2_filename = str(uuid.uuid4()) + "_" + image2_file.filename
            image1_path = os.path.join(app.config['UPLOAD_FOLDER'], image1_filename)
            image2_path = os.path.join(app.config['UPLOAD_FOLDER'], image2_filename)
            image1_file.save(image1_path)
            image2_file.save(image2_path)

            # Process images and get output image path
            change_map_path = process_images(image1_path, image2_path, app.config['UPLOAD_FOLDER'])

            # Extract only the filename from the path
            filename = os.path.basename(change_map_path)
            image1 = image1_filename
            image2 = image2_filename

    return render_template('index.html', image1=image1, image2=image2, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
