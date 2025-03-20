from PyQt5.QtGui import QImage, QPixmap
import cv2
class ImageQT:
    @staticmethod
    def addToQT(image, frame, size):
        if image is not None:
            image_tmp = image.copy()
            image_tmp = cv2.resize(image_tmp, size)
            image_qt = QImage(image_tmp.data, image_tmp.shape[1], image_tmp.shape[0],
                              QImage.Format_RGB888).rgbSwapped()
            pix = QPixmap.fromImage(image_qt).scaled(size[0], size[1], aspectRatioMode=1)
            frame.setPixmap(pix)