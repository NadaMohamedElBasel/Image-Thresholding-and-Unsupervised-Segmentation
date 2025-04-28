from PyQt5.QtWidgets import QApplication,QSpinBox,QDoubleSpinBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox, QGridLayout,QGraphicsView,QFileDialog,QGraphicsScene,QGraphicsPixmapItem
from PyQt5.QtCore import Qt
import sys
import cv2
from PyQt5.QtGui import QPixmap,QImage,QPainter, QPen
import numpy as np
import scipy

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
        
        self.thresholdm2_slider = QSlider(Qt.Horizontal)
        self.thresholdm2_slider.setRange(0, 255)
        self.thresholdm2_value = QLabel("0")
        self.thresholdm2_slider.valueChanged.connect(lambda: self.thresholdm2_value.setText(str(self.thresholdm2_slider.value())))

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

        threshold_layout.addWidget(QLabel("Threshold 2:"), 2, 0)
        threshold_layout.addWidget(self.thresholdm2_slider, 2, 1, 1, 2)  # Slider spans 2 columns
        threshold_layout.addWidget(self.thresholdm2_value, 2, 3)  # Move value to column 3

        threshold_layout.addWidget(self.spin_block_size, 3, 1, 1, 1)  # Move below threshold sliders
        threshold_layout.addWidget(self.spin_offset, 3, 2,1,1)  # Align offset next to spin box
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(self.load_image_button)
        layout.addWidget(threshold_group)
        
        # Segmentation Controls
        segmentation_group = QGroupBox("Segmentation")
        segmentation_layout = QGridLayout()
        
        self.kmeans_button = QPushButton("K-Means")
        self.mean_shift_button = QPushButton("Mean Shift")
        self.agglomerative_button = QPushButton("Agglomerative")
        self.region_growing_button = QPushButton("Region Growing")
        
        self.iterations_slider = QSlider(Qt.Horizontal)
        self.iterations_slider.setRange(1, 100)
        self.iterations_value = QLabel("1")
        self.iterations_slider.valueChanged.connect(lambda: self.iterations_value.setText(str(self.iterations_slider.value())))
        
        self.clusters_slider = QSlider(Qt.Horizontal)
        self.clusters_slider.setRange(1, 10)
        self.clusters_value = QLabel("1")
        self.clusters_slider.valueChanged.connect(lambda: self.clusters_value.setText(str(self.clusters_slider.value())))
        
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

    def optimal(self):
        if self.img is None:
            return
        if self.img.ndim == 3:
            img_proc = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            img_proc = self.img.copy()

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

    def spectral(self, image, num_bands=3, max_iterations=5):

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
