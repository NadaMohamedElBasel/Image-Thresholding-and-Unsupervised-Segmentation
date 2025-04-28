from PyQt5.QtWidgets import QApplication,QSpinBox,QDoubleSpinBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox, QGridLayout,QGraphicsView,QFileDialog,QGraphicsScene,QGraphicsPixmapItem
from PyQt5.QtCore import Qt
import sys
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap,QImage,QPainter, QPen
import numpy as np
import scipy
from sklearn.cluster import MeanShift,AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from skimage.io import imread, imsave

class CVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Image Viewports
        self.input_label = QLabel("Input Image")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_view = QGraphicsView()
        self.output_label = QLabel("Output Image")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_view = QGraphicsView()
        
        image_layout = QHBoxLayout()
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_view)
        
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_view)
        
        image_layout.addLayout(input_layout)
        image_layout.addLayout(output_layout)
        
        layout.addLayout(image_layout)
        # Load Image Button
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        # Thresholding Controls
        threshold_group = QGroupBox("Thresholding")
        threshold_layout = QGridLayout()
        
        self.otsu_button = QPushButton("OTSU")
        self.otsu_button.clicked.connect(self.otsu)
        self.optimal_button = QPushButton("Optimal")
        self.optimal_button.clicked.connect(self.optimal)
        self.spectral_button = QPushButton("Spectral")
        self.spectral_button.clicked.connect(self.spectral)
        self.local_button = QPushButton("Local")
        self.local_button.clicked.connect(self.local)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_value = QLabel("0")
        self.threshold_slider.valueChanged.connect(lambda: self.threshold_value.setText(str(self.threshold_slider.value())))
        
        # self.thresholdm2_slider = QSlider(Qt.Horizontal)
        # self.thresholdm2_slider.setRange(0, 255)
        # self.thresholdm2_value = QLabel("0")
        # self.thresholdm2_slider.valueChanged.connect(lambda: self.thresholdm2_value.setText(str(self.thresholdm2_slider.value())))

        # Local threshold parameters: block size and offset
        self.spin_block_size = QSpinBox()
        self.spin_block_size.setRange(3, 51)
        self.spin_block_size.setSingleStep(2)
        self.spin_block_size.setValue(11)
        self.spin_block_size.setSuffix(" block size (local)")
        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(-50, 50)
        self.spin_offset.setValue(2)
        self.spin_offset.setSuffix(" offset (local)")
    
        threshold_layout.addWidget(self.otsu_button, 0, 0, 1, 1)
        threshold_layout.addWidget(self.optimal_button, 0, 1, 1, 1)
        threshold_layout.addWidget(self.spectral_button, 0, 2, 1, 1)
        threshold_layout.addWidget(self.local_button, 0, 3, 1, 1)

        threshold_layout.addWidget(QLabel("Threshold:"), 1, 0)
        threshold_layout.addWidget(self.threshold_slider, 1, 1, 1, 2)  # Slider spans 2 columns
        threshold_layout.addWidget(self.threshold_value, 1, 3)  # Move value to column 3

        # threshold_layout.addWidget(QLabel("Threshold 2:"), 2, 0)
        # threshold_layout.addWidget(self.thresholdm2_slider, 2, 1, 1, 2)  # Slider spans 2 columns
        # threshold_layout.addWidget(self.thresholdm2_value, 2, 3)  # Move value to column 3

        threshold_layout.addWidget(self.spin_block_size, 3, 1, 1, 1)  # Move below threshold sliders
        threshold_layout.addWidget(self.spin_offset, 3, 2,1,1)  # Align offset next to spin box
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(self.load_image_button)
        layout.addWidget(threshold_group)
        
        # Segmentation Controls
        segmentation_group = QGroupBox("Segmentation")
        segmentation_layout = QGridLayout()
        
        self.kmeans_button = QPushButton("K-Means")
        self.kmeans_button.clicked.connect(self.kmeans)
        self.mean_shift_button = QPushButton("Mean Shift")
        self.mean_shift_button.clicked.connect(self.apply_mean_shift_segmentation)
        self.agglomerative_button = QPushButton("Agglomerative")
        self.agglomerative_button.clicked.connect(self.apply_agglomerative_clustering)
        self.region_growing_button = QPushButton("Region Growing")
        self.region_growing_button.clicked.connect(self.region_growing)
        
        self.iterations_slider = QSlider(Qt.Horizontal)
        self.iterations_slider.setRange(1, 100)
        self.iterations_value = QLabel("5")
        self.iterations_slider.valueChanged.connect(lambda: self.iterations_value.setText(str(self.iterations_slider.value())))
        self.iterations_slider.setValue(5)  # Default value
        
        self.clusters_slider = QSlider(Qt.Horizontal)
        self.clusters_slider.setRange(1, 10)
        self.clusters_value = QLabel("3")
        self.clusters_slider.valueChanged.connect(lambda: self.clusters_value.setText(str(self.clusters_slider.value())))
        self.clusters_slider.setValue(3)
        
        segmentation_layout.addWidget(self.kmeans_button, 0, 0)
        segmentation_layout.addWidget(self.mean_shift_button, 0, 1)
        segmentation_layout.addWidget(self.agglomerative_button, 0, 2)
        segmentation_layout.addWidget(self.region_growing_button, 0, 3)
        segmentation_layout.addWidget(QLabel("Iterations:"), 1, 0)
        segmentation_layout.addWidget(self.iterations_slider, 1, 1, 1, 2)
        segmentation_layout.addWidget(self.iterations_value, 1, 3)
        segmentation_layout.addWidget(QLabel("Clusters:"), 2, 0)
        segmentation_layout.addWidget(self.clusters_slider, 2, 1, 1, 2)
        segmentation_layout.addWidget(self.clusters_value, 2, 3)
        
        segmentation_group.setLayout(segmentation_layout)
        layout.addWidget(segmentation_group)
        
        self.setLayout(layout)
        self.setWindowTitle("Computer Vision Task UI")
        self.setGeometry(100, 100, 800, 600)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        # Read and process image
        img = cv2.imread(file_path)
        self.img=img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for proper display

        # Convert OpenCV image (NumPy array) to QImage
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        q_pixmap = QPixmap.fromImage(q_image)

        # Display image in QGraphicsView
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.input_view.setScene(scene)


    # def apply_mean_shift_segmentation(self):
    #     if not hasattr(self, 'img') or self.img is None:
    #         print("No image loaded!")
    #         return

    #     # Convert to RGB if needed
    #     image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
    #     # Convert to LAB
    #     lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    #     height, width, _ = image.shape

    #     # Flatten LAB image
    #     flat_image = lab_image.reshape((-1, 3))

    #     # Generate x, y coordinates
    #     x, y = np.meshgrid(np.arange(width), np.arange(height))
    #     x = (x / width) * 255  # Normalize x
    #     y = (y / height) * 255 # Normalize y

    #     # Stack LAB + normalized spatial coordinates
    #     flat_image_with_coordinates = np.column_stack([flat_image, x.flatten(), y.flatten()])

    #     # Estimate bandwidth
    #     bandwidth = estimate_bandwidth(flat_image_with_coordinates, quantile=0.1, n_samples=1000)

    #     # Mean Shift
    #     mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #     mean_shift.fit(flat_image_with_coordinates)

    #     # Get labels
    #     labels = mean_shift.labels_
    #     segmented_image = labels.reshape((height, width))

    #     # Color segmented image
    #     unique_labels = np.unique(labels)
    #     segmented_colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
    #     colored_segmented_image = segmented_colors[segmented_image]

    #     # Prepare for QImage display
    #     colored_segmented_image = np.clip(colored_segmented_image, 0, 255).astype(np.uint8)
    #     height, width, _ = colored_segmented_image.shape
    #     bytes_per_line = 3 * width
    #     q_image = QImage(colored_segmented_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

    #     # Display
    #     q_pixmap = QPixmap.fromImage(q_image)
    #     scene = QGraphicsScene()
    #     scene.addItem(QGraphicsPixmapItem(q_pixmap))
    #     self.output_view.setScene(scene)

    # def apply_mean_shift_segmentation(self, spatial_radius=6, color_radius=4.5, max_iter=300):
    #     """
    #     Apply Mean Shift segmentation to an image.
        
    #     Parameters:
    #     - image: Input image (H, W, 3) in uint8
    #     - spatial_radius: The spatial window radius.
    #     - color_radius: The color window radius.
    #     - max_iter: Maximum number of iterations for convergence.

    #     Returns:
    #     - segmented_image: Output segmented image
    #     """
    #     # Resize for faster processing if needed (optional)
    #     # image = cv2.resize(image, (width, height))
    #     if not hasattr(self, 'img') or self.img is None:
    #         print("No image loaded!")
    #         return
    #     # Convert to RGB if needed
    #     image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
    #     # Convert image to (N, 5) where N = H*W, features = (r, g, b, x, y)
    #     h, w, c = image.shape
    #     X = np.zeros((h * w, 5), dtype=np.float32)
    #     for i in range(h):
    #         for j in range(w):
    #             r, g, b = image[i, j]
    #             X[i * w + j] = np.array([r, g, b, spatial_radius * i / h, spatial_radius * j / w])

    #     # Estimate bandwidth
    #     bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=500)
    #     if bandwidth <= 0:
    #         bandwidth = 1  # Avoid zero bandwidth error
    #     # MeanShift clustering
    #     ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=max_iter)
    #     ms.fit(X)
    #     labels = ms.labels_
    #     cluster_centers = ms.cluster_centers_
    #     # Build the segmented image
    #     segmented_image = np.zeros((h, w, 3), dtype=np.uint8)
    #     for i in range(h):
    #         for j in range(w):
    #             label = labels[i * w + j]
    #             r, g, b, _, _ = cluster_centers[label]
    #             segmented_image[i, j] = [int(r), int(g), int(b)]

    #         # Prepare for QImage display
    #     segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)
    #     height, width, _ = segmented_image.shape
    #     bytes_per_line = 3 * width
    #     q_image = QImage(segmented_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    #     # Display
    #     q_pixmap = QPixmap.fromImage(q_image)
    #     scene = QGraphicsScene()
    #     scene.addItem(QGraphicsPixmapItem(q_pixmap))
    #     self.output_view.setScene(scene)

    #     return segmented_image

    def apply_mean_shift_segmentation(self):
        if not hasattr(self, 'img') or self.img is None:
            print("No image loaded!")
            return

        # Parameters
        bandwidth = 30  # Bandwidth radius for shifting (you can tune this)
        #max_iterations = 5
        max_iterations = self.iterations_slider.value()  # Get from slider
        convergence_threshold = 1  # Threshold for mean shift convergence

        # Prepare image
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (256, 256))
        flattened_image = np.copy(resized_image.reshape((-1, 3)))

        shifted_points = np.copy(flattened_image)

        for iteration in range(max_iterations):
            new_points = []
            for point in shifted_points:
                # Find all points within bandwidth
                distances = np.linalg.norm(flattened_image - point, axis=1)
                within_bandwidth = flattened_image[distances < bandwidth]

                if len(within_bandwidth) > 0:
                    # Mean shift: move to the mean of nearby points
                    mean_point = np.mean(within_bandwidth, axis=0)
                    new_points.append(mean_point)
                else:
                    # If no nearby points, stay the same
                    new_points.append(point)

            new_points = np.array(new_points)

            # Check convergence
            shift_distances = np.linalg.norm(new_points - shifted_points, axis=1)
            if np.max(shift_distances) < convergence_threshold:
                print(f"Converged after {iteration+1} iterations.")
                break

            shifted_points = new_points

        # After convergence, assign each pixel to its nearest mode
        final_pixels = []
        for p in shifted_points:
            final_pixels.append(np.round(p))
        final_pixels = np.array(final_pixels, np.uint8)
        output_image = final_pixels.reshape(resized_image.shape)

        # Display the result
        height, width, _ = output_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(output_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)


##############################################################################################################################
    def kmeans(self):
        if self.img is None:
            return
        
        # Get the number of clusters from the slider
        num_clusters = self.clusters_slider.value()
        
        # Get the number of iterations from the slider
        max_iterations = self.iterations_slider.value()
        
        # Prepare the image for KMeans segmentation
        if self.img.ndim == 3:
            # For color images, use all channels and convert from BGR to RGB
            img_for_kmeans = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            # Reshape the image to a 2D array of pixels
            pixels = img_for_kmeans.reshape((-1, 3))
        else:
            # For grayscale images
            img_for_kmeans = self.img.copy()
            # Reshape the image to a 2D array of pixels
            pixels = img_for_kmeans.reshape((-1, 1))
        
        # Convert pixel values to float32 for computations
        pixels = np.float32(pixels)
        n_pixels = pixels.shape[0]
        
        # Initialize centroids by randomly choosing pixels from the image
        indices = np.random.choice(n_pixels, num_clusters, replace=False)
        centroids = pixels[indices]
        
        for i in range(max_iterations):
            # Compute Euclidean distances between pixels and centroids
            # The result is a (n_pixels x num_clusters) array
            distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
            # Assign each pixel to the closest centroid
            labels = np.argmin(distances, axis=1)
            
            # Recompute centroids as the mean of all pixels assigned to each cluster
            new_centroids = []
            for k in range(num_clusters):
                if np.any(labels == k):
                    new_centroid = pixels[labels == k].mean(axis=0)
                else:
                    # If a centroid loses all assigned pixels, keep the previous value
                    new_centroid = centroids[k]
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids, dtype=np.float32)
            
            # If centroids do not change much, break early
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        
        # Create the segmented image by replacing each pixel with its centroid value
        segmented_pixels = centroids[labels]
        segmented_pixels = np.uint8(segmented_pixels)
        
        if self.img.ndim == 3:
            segmented_img = segmented_pixels.reshape(img_for_kmeans.shape)
            # For display consistency, convert from RGB to BGR and back to RGB
            segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
            display_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
            height, width, channel = display_img.shape
            bytes_per_line = 3 * width
            q_image = QImage(display_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            segmented_img = segmented_pixels.reshape(img_for_kmeans.shape)
            height, width = segmented_img.shape
            bytes_per_line = width  # Grayscale: 1 byte per pixel
            q_image = QImage(segmented_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Display the result
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)
############################################# Agglomerative Clustering #######################################################
    clusters_list = []
    cluster = {}
    centers = {}

    def calculate_distance(self,x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def clusters_average_distance(self,cluster1, cluster2):
    
        cluster1_center = np.average(cluster1)
        cluster2_center = np.average(cluster2)
        return self.calculate_distance(cluster1_center, cluster2_center) 

    def initial_clusters(self,image_clusters):
    
        global initial_k
        groups = {}
        cluster_color = int(256 / initial_k)
        for i in range(initial_k):
            color = i * cluster_color
            groups[(color, color, color)] = []
        for i, p in enumerate(image_clusters):
            go = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))
            groups[go].append(p)
        return [group for group in groups.values() if len(group) > 0]

    def get_cluster_center( self,point):
        global cluster
        point_cluster_num = self.cluster[tuple(point)]
        center = self.centers[point_cluster_num]
        return center

    def get_clusters(self,image_clusters):
        global clusters_list
        clusters_list = self.initial_clusters(image_clusters)

        while len(clusters_list) > clusters_number:
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
                key=lambda c: self.clusters_average_distance(c[0], c[1]))

            clusters_list = [cluster_itr for cluster_itr in clusters_list if cluster_itr != cluster1 and cluster_itr != cluster2]

            merged_cluster = cluster1 + cluster2

            clusters_list.append(merged_cluster)

        global cluster 
        for cl_num, cl in enumerate(clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        global centers 
        for cl_num, cl in enumerate(clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def apply_agglomerative_clustering(self):
        global clusters_number
        global initial_k
        if not hasattr(self, 'img') or self.img is None:
            print("No image loaded!")
            return
        # Convert to RGB if needed
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        initial_number_of_clusters=3
        resized_image = cv2.resize(image, (256,256))

        clusters_number = self.clusters_slider.value()
        initial_k = initial_number_of_clusters 
        flattened_image = np.copy(resized_image.reshape((-1, 3)))

        self.get_clusters(flattened_image)
        output_image = []
        for row in resized_image:
            rows = []
            for col in row:
                rows.append(self.get_cluster_center(list(col)))
            output_image.append(rows)    
        output_image = np.array(output_image, np.uint8)
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        height, width, _ = output_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(output_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Display
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)
#################################################################################################################################################

    def region_growing(self):
        if self.img is None:
            return
        
        # Get the image in grayscale
        if self.img.ndim == 3:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = self.img.copy()
        
        # Get threshold value from slider for similarity criterion
        tolerance = self.threshold_slider.value()
        if tolerance == 0:
            tolerance = 10  # Default tolerance if slider is at 0
        
        # Get the dimensions of the image
        height, width = img_gray.shape
        
        # Initialize segmentation result
        segmented = np.zeros_like(img_gray, dtype=np.uint8)
        visited = np.zeros_like(img_gray, dtype=bool)
        
        # Calculate seed points
        # We'll place seeds in a grid pattern across the image
        grid_size = 30  # Distance between seed points
        region_id = 1
        
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                if not visited[y, x]:
                    # Initialize region growing from this seed
                    region = np.zeros_like(img_gray, dtype=bool)
                    stack = [(y, x)]
                    seed_value = int(img_gray[y, x])
                    
                    while stack:
                        cy, cx = stack.pop()
                        
                        # Skip if outside image bounds or already visited
                        if (cy < 0 or cy >= height or cx < 0 or cx >= width or 
                                visited[cy, cx] or region[cy, cx]):
                            continue
                        
                        # Check if pixel is similar to seed
                        current_value = int(img_gray[cy, cx])
                        if abs(current_value - seed_value) <= tolerance:
                            region[cy, cx] = True
                            visited[cy, cx] = True
                            
                            # Add neighbors to stack
                            stack.append((cy-1, cx))  # Up
                            stack.append((cy+1, cx))  # Down
                            stack.append((cy, cx-1))  # Left
                            stack.append((cy, cx+1))  # Right
                    
                    # Add the region to the segmentation result
                    if np.any(region):
                        segmented[region] = region_id
                        region_id += 1
        
        # Scale to 0-255 for better visualization
        if np.max(segmented) > 0:
            scale_factor = 255.0 / np.max(segmented)
            segmented = (segmented * scale_factor).astype(np.uint8)
        
        # Apply a random color to each region for better visualization
        if region_id > 1:
            # Create a colormap
            colormap = np.zeros((region_id, 3), dtype=np.uint8)
            # First region is background (black)
            colormap[1:] = np.random.randint(0, 255, size=(region_id-1, 3))
            
            # Create colored segmentation
            colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(1, region_id):
                colored_segmentation[segmented == int(i * scale_factor)] = colormap[i]
            
            # Convert to QImage for display
            colored_rgb = cv2.cvtColor(colored_segmentation, cv2.COLOR_BGR2RGB)
            q_image = QImage(colored_rgb.data, width, height, 3 * width, QImage.Format_RGB888)
        else:
            # Grayscale output if no regions found
            q_image = QImage(segmented.data, width, height, width, QImage.Format_Grayscale8)
        
        # Convert QImage to QPixmap and display
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)
####################################################################################################################################################
    def optimal(self):
        if self.img is None:
            return
        if self.img.ndim == 3:
            img_proc = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            img_proc = cv2.resize(img_proc, (256,256))
        else:
            img_proc = self.img.copy()
            img_proc = cv2.resize(img_proc, (256,256))

        cornerSize = 10  # Corner size

        top_left = img_proc[:cornerSize, :cornerSize]
        top_right = img_proc[:cornerSize, -cornerSize:]
        bottom_left = img_proc[-cornerSize:, :cornerSize]
        bottom_right = img_proc[-cornerSize:, -cornerSize:]

        corners = np.hstack((top_left.ravel(), top_right.ravel(),
                             bottom_left.ravel(), bottom_right.ravel()))

        threshold = np.mean(corners)

        img = img_proc
        for _ in range(100):
            background = img[img <= threshold]
            object = img[img > threshold]

            if len(background) == 0 or len(object) == 0:
                break  # Avoid division by zero

            # Compute means
            mu_b = np.mean(background)
            mu_o = np.mean(object)

            # Update threshold
            threshold_new = (mu_b + mu_o) / 2

            # Check for convergence
            if abs(threshold - threshold_new) < 0.1:
                break
            threshold = threshold_new

        thresh_img = (img_proc > threshold).astype(np.uint8) * 255

        # Convert NumPy array to QImage
        height, width = thresh_img.shape
        bytes_per_line = width  # Since it's grayscale, 1 byte per pixel
        q_image = QImage(thresh_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap and display
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)


##################################################################################################################################
    def otsu(self):
        if self.img is None:
            return
        if self.img.ndim == 3:
            img_proc = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            img_proc = self.img.copy()

        hist, _ = np.histogram(img_proc.flatten(), bins=256, range=[0,256])
        total = img_proc.size
        sum_total = np.dot(np.arange(256), hist)
        current_max, threshold = 0, 0
        sumB, wB = 0, 0
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            between_var = wB * wF * (mB - mF) ** 2
            if between_var > current_max:
                current_max = between_var
                threshold = t
        thresh_img = (img_proc > threshold).astype(np.uint8) * 255

        # Convert NumPy array to QImage
        height, width = thresh_img.shape
        bytes_per_line = width  # Since it's grayscale, 1 byte per pixel
        q_image = QImage(thresh_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap and display
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)
###########################################################################################################################################
    def spectral(self):
        num_bands=3
        #max_iterations=5
        max_iterations = self.iterations_slider.value()  # Get from slider
        image=self.img
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create multiple spectral bands from the input image
        bands = [image]
        for i in range(1, num_bands):
            kernel_size = 2 * i + 1  # Ensure odd kernel size
            bands.append(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0))

        # Initialize the whole image as a single region
        height, width = image.shape
        segmentation_result = np.zeros((height, width), dtype=np.uint8)
        regions = [np.ones((height, width), dtype=bool)]
        region_id = 1

        # Process each iteration
        for iteration in range(max_iterations):
            new_regions = []

            if not regions:  # If no regions left, we're done
                break

            for region_mask in regions:
                should_segment_further = True

                # Process each band
                sub_regions_per_band = []
                for band in bands:
                    # Get region pixels
                    region_pixels = band[region_mask]

                    if len(region_pixels) == 0:
                        should_segment_further = False
                        break

                    # Compute histogram
                    hist = cv2.calcHist([region_pixels], [0], None, [256], [0, 256])
                    hist = hist.flatten() / len(region_pixels)  # Normalize

                    # Apply Gaussian smoothing to histogram
                    hist_smoothed = cv2.GaussianBlur(hist, (5, 1), 0)

                    # Find peaks (local maxima)
                    peaks = []
                    for i in range(1, len(hist_smoothed) - 1):
                        if hist_smoothed[i] > hist_smoothed[i - 1] and hist_smoothed[i] > hist_smoothed[i + 1]:
                            peaks.append((i, hist_smoothed[i]))

                    # If no peaks or only one peak, can't segment further
                    if len(peaks) <= 1:
                        should_segment_further = False
                        break

                    # Find most significant peak
                    significant_peak_idx = max(peaks, key=lambda x: x[1])[0]

                    # Find local minima on either side of the peak
                    left_min_idx = None
                    for i in range(significant_peak_idx, 0, -1):
                        if hist_smoothed[i] < hist_smoothed[i - 1] and hist_smoothed[i] < hist_smoothed[i + 1]:
                            left_min_idx = i
                            break

                    right_min_idx = None
                    for i in range(significant_peak_idx, len(hist_smoothed) - 1):
                        if hist_smoothed[i] < hist_smoothed[i - 1] and hist_smoothed[i] < hist_smoothed[i + 1]:
                            right_min_idx = i
                            break

                    # If couldn't find minima, can't segment further
                    if left_min_idx is None or right_min_idx is None:
                        should_segment_further = False
                        break

                    # Create new sub-regions
                    left_mask = np.zeros_like(region_mask, dtype=bool)
                    middle_mask = np.zeros_like(region_mask, dtype=bool)
                    right_mask = np.zeros_like(region_mask, dtype=bool)

                    # Apply thresholds
                    region_indices = np.where(region_mask)
                    for i in range(len(region_indices[0])):
                        y, x = region_indices[0][i], region_indices[1][i]
                        pixel_value = band[y, x]

                        if pixel_value < left_min_idx:
                            left_mask[y, x] = True
                        elif pixel_value > right_min_idx:
                            right_mask[y, x] = True
                        else:
                            middle_mask[y, x] = True

                    # Add sub-regions to the list
                    band_regions = [left_mask, middle_mask, right_mask]
                    sub_regions_per_band.append(band_regions)

                # If no further segmentation is possible, mark current region with ID
                if not should_segment_further:
                    segmentation_result[region_mask] = region_id
                    region_id += 1
                    continue

                # Combine sub-regions from different bands
                # Start with the first band's regions
                combined_regions = sub_regions_per_band[0]

                # Intersect with regions from other bands
                for band_idx in range(1, len(bands)):
                    new_combined_regions = []
                    for region1 in combined_regions:
                        for region2 in sub_regions_per_band[band_idx]:
                            intersection = np.logical_and(region1, region2)
                            if np.any(intersection):
                                new_combined_regions.append(intersection)
                    combined_regions = new_combined_regions

                # Add combined regions to the list for next iteration
                for region in combined_regions:
                    if np.any(region):
                        new_regions.append(region)

            # If no new regions were created, we're done
            if not new_regions:
                break

            regions = new_regions

        # Label any remaining regions
        for region_mask in regions:
            segmentation_result[region_mask] = region_id
            region_id += 1

        # Scale to 0-255 for display
        if np.max(segmentation_result) > 0:
            # Calculate scaling factor to distribute values evenly across 0-255
            scale_factor = 255.0 / np.max(segmentation_result)
            segmentation_result = (segmentation_result * scale_factor).astype(np.uint8)

            thresh_img = segmentation_result
            height, width = thresh_img.shape
            bytes_per_line = width  # Since it's grayscale, 1 byte per pixel
            q_image = QImage(thresh_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            # Convert QImage to QPixmap and display
            q_pixmap = QPixmap.fromImage(q_image)
            scene = QGraphicsScene()
            scene.addItem(QGraphicsPixmapItem(q_pixmap))
            self.output_view.setScene(scene)
##############################################################################################################################################################
    def local(self):# adaptive mean thresholding
        if self.img is None:
            return
        if self.img.ndim == 3:
            img_proc = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            img_proc = self.img.copy()
        block_size = self.spin_block_size.value()
        if block_size % 2 == 0:
            block_size += 1
        offset = self.spin_offset.value()
        kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size**2)
        local_mean = scipy.signal.convolve2d(img_proc.astype(np.float32), kernel, mode='same', boundary='symm')
        thresh_img = (img_proc > (local_mean - offset)).astype(np.uint8) * 255
        # Convert NumPy array to QImage
        height, width = thresh_img.shape
        bytes_per_line = width  # Since it's grayscale, 1 byte per pixel
        q_image = QImage(thresh_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap and display
        q_pixmap = QPixmap.fromImage(q_image)
        scene = QGraphicsScene()
        scene.addItem(QGraphicsPixmapItem(q_pixmap))
        self.output_view.setScene(scene)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CVApp()
    window.show()
    sys.exit(app.exec_())
