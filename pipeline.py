import os
import cv2
import numpy as np
import textwrap
import re

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from ultralytics import YOLO
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator


class MangaTranslatorPipeline:
    """
    Pipeline:
    YOLOv11
        ↓
    MangaOCR
        ↓
    Google Translator
        ↓
    OpenCV Inpainting
        ↓
    Render Translation
    """

    def __init__(
        self,
        yolo_weights_path="yolo11s.pt",
        comic_font_path=None,
        target_language="id"
    ):

        print("Loading YOLO...")
        self.yolo_model = YOLO(yolo_weights_path)

        print("Loading MangaOCR...")
        self.mocr = MangaOcr()

        print("Loading Translator...")
        self.translator = GoogleTranslator(
            source="ja",
            target=target_language
        )

        self.font_path = comic_font_path
        self.target_language = target_language

    ###########################################################
    # Remove Nested
    ###########################################################
    def remove_nested_boxes(self, boxes, overlap_thresh=0.9):
        """
        boxes = [(x1, y1, x2, y2, conf), ...]
        """

        # Urutkan berdasarkan luas (terbesar dulu)
        boxes = sorted(
            boxes,
            key=lambda b: (b[2]-b[0]) * (b[3]-b[1]),
            reverse=True
        )

        keep = []

        for box in boxes:
            x1, y1, x2, y2, conf = box
            area_small = max((x2 - x1) * (y2 - y1), 1)

            inside = False

            for kept in keep:
                kx1, ky1, kx2, ky2, _ = kept

                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                inter = (ix2-ix1) * (iy2-iy1)

                if inter / area_small > overlap_thresh:
                    inside = True
                    break

            if not inside:
                keep.append(box)

        return keep

    ###########################################################
    # Bubble Detection
    ###########################################################

    def detect_bubbles(
            self,
            img_path,
            conf_threshold=0.30
        ):


            results = self.yolo_model(
                img_path,
                conf=conf_threshold
            )[0]

            if len(results.boxes) == 0:
                return []   

            boxes = []

            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, conf in zip(xyxy, confs):

                x1, y1, x2, y2 = map(int, box)

                boxes.append((x1, y1, x2, y2, float(conf)))

            # Hapus box yang berada di dalam box lain
            boxes = self.remove_nested_boxes(boxes)

            # Urutkan dari atas ke bawah, kiri ke kanan
            boxes = sorted(
                boxes,
                key=lambda b: (b[1], b[0])
            )

            # Buang nilai confidence karena tidak dipakai lagi
            boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in boxes]

            return boxes

    ###########################################################
    # Clean Bubble
    ###########################################################

    def clean_screentone_bubble(
        self,
        image_cv,
        box,
        expansion_pixels=3
    ):

        x1, y1, x2, y2 = box

        bubble_crop = image_cv[y1:y2, x1:x2]

        if bubble_crop.size == 0:
            return image_cv

        gray = cv2.cvtColor(
            bubble_crop,
            cv2.COLOR_BGR2GRAY
        )

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones(
            (
                expansion_pixels,
                expansion_pixels
            ),
            np.uint8
        )

        mask = cv2.dilate(
            thresh,
            kernel,
            iterations=1
        )

        cleaned_crop = cv2.inpaint(
            bubble_crop,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )

        cleaned_image = image_cv.copy()

        cleaned_image[
            y1:y2,
            x1:x2
        ] = cleaned_crop

        return cleaned_image

    ##########################################################
    # OCR
    ##########################################################

    def run_ocr(
        self,
        bubble_crop
    ):

        try:

            text = self.mocr(bubble_crop)

            if text is None:
                return ""

            return text.strip()

        except Exception as e:

            print("OCR Error :", e)

            return ""

    ####################################################################
    # Translation
    ####################################################################

    def translate_text(
        self,
        japanese_text
    ):

        if japanese_text == "":
            return ""

        try:

            translated = self.translator.translate(
                japanese_text
            )

            return translated

        except Exception as e:

            print("Translate Error :", e)

            return japanese_text

    ####################################################################
    # Utility
    ####################################################################

    def load_font(self, size):

        print("================================")
        print("FONT PATH :", self.font_path)
        print(
            "FONT EXISTS :",
            os.path.exists(self.font_path)
            if self.font_path
            else False
        )
        print("REQUESTED SIZE :", size)
    
        try:
    
            if self.font_path:
    
                font = ImageFont.truetype(
                    self.font_path,
                    size
                )
    
                print("FONT LOADED SUCCESSFULLY")
                print("ACTUAL FONT SIZE :", font.size)
    
                return font
    
        except Exception as e:
    
            print("FONT ERROR :", e)
    
        print("USING DEFAULT PIL FONT")
    
        return ImageFont.load_default()
    

    def get_text_colors(self, image_pil, box):
        x1, y1, x2, y2 = box

        crop = np.array(image_pil.crop((x1, y1, x2, y2)))

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        # ============================
        # Ambil hanya area tengah bubble
        # ============================
        h, w = gray.shape

        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)

        center = gray[
            margin_y:h-margin_y,
            margin_x:w-margin_x
        ]

        mean = center.mean()

        if mean > 150:
            return (0, 0, 0), (255, 255, 255)   # hitam, outline putih
        else:
            return (255, 255, 255), (0, 0, 0)   # putih, outline hitam

    ####################################################################
    # Auto Scale Text
    ####################################################################

    def auto_scale_manga_text(
        self,
        text,
        box,
        max_font_size=42,
        min_font_size=14,
        padding=12
    ):

        x1, y1, x2, y2 = box

        box_width = x2 - x1
        box_height = y2 - y1

        usable_width = max(
            10,
            int((box_width - (padding * 2)) * 0.80)
        )

        usable_height = max(
            10,
            box_height - (padding * 2)
        )

        best_font = self.load_font(min_font_size)
        best_text = text
        best_height = 0

        for font_size in range(
            max_font_size,
            min_font_size - 1,
            -1
        ):

            font = self.load_font(font_size)

            bbox = font.getbbox("A")

            char_width = max(
                1,
                bbox[2] - bbox[0]
            )

            chars_per_line = max(
                1,
                usable_width // char_width
            )

            wrapped = textwrap.fill(
                text,
                width=chars_per_line
            )

            lines = wrapped.split("\n")

            total_height = 0
            max_width = 0

            line_spacing = max(
                2,int(font_size * 0.30)
            )

            for line in lines:

                bbox = font.getbbox(line)

                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                max_width = max(
                    max_width,
                    w
                )

                total_height += h + line_spacing

            total_height -= line_spacing

            if (
                max_width <= usable_width and
                total_height <= usable_height
            ):

                best_font = font
                best_text = wrapped
                best_height = total_height

                break

        return (
            best_font,
            best_text,
            best_height
        )

    ####################################################################
    # Render Translation
    ####################################################################

    def render_text_to_bubble(
        self,
        image_pil,
        text,
        box,
        font_color=(0, 0, 0)
    ):

        draw = ImageDraw.Draw(image_pil)

        font_color, stroke_color = self.get_text_colors(
            image_pil,
            box
        )

        x1, y1, x2, y2 = box

        box_width = x2 - x1
        box_height = y2 - y1

        (
            font,
            wrapped_text,
            text_height
        ) = self.auto_scale_manga_text(
            text=text,
            box=box
        )

        current_y = y1 + max(
            2,
            (box_height - text_height) // 2
        )

        try:
            font_size = font.size
        except:
            font_size = 12

        line_spacing = max(
            2,
            int(font_size * 0.20)
        )

        lines = wrapped_text.split("\n")

        for line in lines:

            bbox = font.getbbox(line)

            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]

            current_x = x1 + max(
                2,
                (box_width - line_width) // 2
            )

            draw.text(
                (current_x, current_y),
                line,
                font=font,
                fill=font_color,

                stroke_width=2,
                stroke_fill=stroke_color
            )

            current_y += (
                line_height +
                line_spacing
            )

        return image_pil
    ##########################################################
    # GRAYSCALE
    ##########################################################
    def normalize_image(self, path):

        img = (
            Image.open(path)
            .convert("L")      # grayscale
            .convert("RGB")    # 3 channel
        )

        img.save(path)

    ####################################################################
    # Main Pipeline
    ####################################################################

    def process_manga_page(
        self,
        input_image_path,
        output_image_path,
        conf_threshold=0.30
    ):

        if not os.path.exists(input_image_path):
            raise FileNotFoundError(input_image_path)

        if os.path.getsize(input_image_path) == 0:
            raise ValueError("Uploaded image is empty.")

          # Ubah gambar menjadi grayscale
        self.normalize_image(input_image_path)

        image_pil_original = Image.open(input_image_path).convert("RGB")

        image_cv = cv2.cvtColor(
            np.array(image_pil_original),
            cv2.COLOR_RGB2BGR
        )

        boxes = self.detect_bubbles(
            input_image_path,
            conf_threshold
        )

        translations = []
        ocr_results = []

        ###########################################################
        # OCR + Translation
        ###########################################################

        for box in boxes:

            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            if width < 30 or height < 20:

                translations.append(None)

                ocr_results.append("")

                continue

            bubble_crop = image_pil_original.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            #######################################################
            # OCR
            #######################################################
            def clean_ocr_text(text):
                text = text.replace("．", ".")
                text = re.sub(r"\.{2,}", "...", text)
                text = re.sub(r"\s+", " ", text)
                return text.strip()

            japanese_text = self.run_ocr(bubble_crop)
            text = clean_ocr_text(japanese_text)

            ocr_results.append(text)

            if japanese_text == "":
                translations.append(None)
                continue

            if not re.search(r"[ぁ-んァ-ン一-龯A-Za-z0-9]", japanese_text):
                translations.append(None)
                continue


            #######################################################
            # Translation
            #######################################################

            translated = self.translate_text(
                text
            )

            translations.append(
                translated
            )

        ###########################################################
        # Remove Original Text
        ###########################################################

        for i, box in enumerate(boxes):

            if translations[i] is not None:

                image_cv = self.clean_screentone_bubble(
                    image_cv,
                    box
                )

        ###########################################################
        # OpenCV -> PIL
        ###########################################################

        final_image = Image.fromarray(

            cv2.cvtColor(
                image_cv,
                cv2.COLOR_BGR2RGB
            )

        )

        ###########################################################
        # Draw Translation
        ###########################################################

        for i, box in enumerate(boxes):

            if translations[i] is None:
                continue

            final_image = self.render_text_to_bubble(
                final_image,
                translations[i],
                box
            )

        ###########################################################
        # Save
        ###########################################################

        output_dir = os.path.dirname(
            output_image_path
        )

        if output_dir != "":

            os.makedirs(
                output_dir,
                exist_ok=True
            )

        final_image.save(
            output_image_path
        )

        ###########################################################
        # Return
        ###########################################################

        return {

            "boxes": boxes,

            "ocr": ocr_results,

            "translation": translations,

            "output": output_image_path

        }
