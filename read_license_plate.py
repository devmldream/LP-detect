import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        return text, score

    return None, None
