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
        self.spectral_button = QPushButton("Spectral")
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
