from PIL import (
    Image,
    ImageEnhance,
    ImageOps,
    ImageFilter,
    ImageDraw,
    ImageFont
)

import numpy as np
import cv2
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_IMAGE = "assets/portrait.jpg"
OUTPUT_IMAGE = "assets/portrait_ascii.png"

# Dark -> bright ASCII characters
ASCII_CHARS = "@%#*+=-:. "

# ASCII width
ASCII_WIDTH = 110

# Image enhancement
CONTRAST = 1.35
BRIGHTNESS = 1.05
SHARPNESS = 1.4

# Character size
SCALE = 8

# ASCII character height correction
ASPECT_RATIO_CORRECTION = 0.48


# ============================================================
# CHECK INPUT IMAGE
# ============================================================

if not os.path.exists(INPUT_IMAGE):
    raise FileNotFoundError(
        f"\nInput image not found:\n{INPUT_IMAGE}\n\n"
        "Make sure portrait.jpg is inside the assets folder."
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(INPUT_IMAGE).convert("RGB")

print("Original image size:", image.size)


# ============================================================
# CROP IMAGE
# ============================================================

width, height = image.size

crop_ratio = 0.90

new_width = int(width * crop_ratio)
new_height = int(height * crop_ratio)

left = (width - new_width) // 2
top = (height - new_height) // 2

right = left + new_width
bottom = top + new_height

image = image.crop(
    (left, top, right, bottom)
)


# ============================================================
# GRAYSCALE
# ============================================================

gray = ImageOps.grayscale(image)


# ============================================================
# CONTRAST
# ============================================================

gray = ImageEnhance.Contrast(
    gray
).enhance(CONTRAST)


# ============================================================
# BRIGHTNESS
# ============================================================

gray = ImageEnhance.Brightness(
    gray
).enhance(BRIGHTNESS)


# ============================================================
# SHARPNESS
# ============================================================

gray = ImageEnhance.Sharpness(
    gray
).enhance(SHARPNESS)


# ============================================================
# DENOISE USING OPENCV
# ============================================================

gray_np = np.array(gray)

gray_np = cv2.bilateralFilter(
    gray_np,
    d=7,
    sigmaColor=50,
    sigmaSpace=50
)

gray = Image.fromarray(
    gray_np
)


# ============================================================
# CLAHE CONTRAST ENHANCEMENT
# ============================================================

gray_np = np.array(gray)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

gray_np = clahe.apply(
    gray_np
)

gray = Image.fromarray(
    gray_np
)


# ============================================================
# RESIZE FOR ASCII
# ============================================================

width, height = gray.size

ascii_width = ASCII_WIDTH

ascii_height = int(
    (height / width)
    * ascii_width
    * ASPECT_RATIO_CORRECTION
)

if ascii_height < 1:
    ascii_height = 1


gray = gray.resize(
    (
        ascii_width,
        ascii_height
    ),
    Image.Resampling.LANCZOS
)


# ============================================================
# SHARPEN AFTER RESIZE
# ============================================================

gray = gray.filter(
    ImageFilter.UnsharpMask(
        radius=1.2,
        percent=120,
        threshold=3
    )
)


# ============================================================
# CONVERT IMAGE TO ASCII
# ============================================================

pixels = np.array(gray)

ascii_length = len(
    ASCII_CHARS
)

ascii_rows = []


for row in pixels:

    line = ""

    for pixel in row:

        normalized = pixel / 255.0

        index = int(
            normalized
            * (ascii_length - 1)
        )

        index = max(
            0,
            min(
                index,
                ascii_length - 1
            )
        )

        line += ASCII_CHARS[index]

    ascii_rows.append(line)


# ============================================================
# CREATE OUTPUT IMAGE
# ============================================================

char_width = SCALE
char_height = int(
    SCALE * 1.7
)

output_width = (
    ascii_width
    * char_width
)

output_height = (
    ascii_height
    * char_height
)

ascii_image = Image.new(
    "RGB",
    (
        output_width,
        output_height
    ),
    (0, 0, 0)
)


# ============================================================
# LOAD MONOSPACE FONT
# ============================================================

font_size = int(
    SCALE * 1.55
)

font = None


# Windows Consolas
font_paths = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/lucon.ttf"
]


for font_path in font_paths:

    if os.path.exists(font_path):

        try:

            font = ImageFont.truetype(
                font_path,
                font_size
            )

            print(
                "Font loaded:",
                font_path
            )

            break

        except Exception:
            pass


# Fallback font
if font is None:

    print(
        "Warning: Windows monospace font "
        "could not be loaded."
    )

    font = ImageFont.load_default()


# ============================================================
# DRAW ASCII CHARACTERS
# ============================================================

draw = ImageDraw.Draw(
    ascii_image
)


for y, line in enumerate(
    ascii_rows
):

    for x, char in enumerate(
        line
    ):

        pixel_value = pixels[y, x]

        brightness = int(
            30
            + (pixel_value / 255)
            * 225
        )

        brightness = max(
            0,
            min(
                brightness,
                255
            )
        )

        text_color = (
            brightness,
            brightness,
            brightness
        )

        draw.text(
            (
                x * char_width,
                y * char_height
            ),
            char,
            font=font,
            fill=text_color
        )


# ============================================================
# SAVE OUTPUT
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_IMAGE),
    exist_ok=True
)

ascii_image.save(
    OUTPUT_IMAGE,
    "PNG"
)


# ============================================================
# TERMINAL PREVIEW
# ============================================================

print()
print("=" * 60)
print("ASCII PREVIEW")
print("=" * 60)
print()

for line in ascii_rows:
    print(line)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print()
print("Input:")
print(INPUT_IMAGE)

print()
print("Output:")
print(OUTPUT_IMAGE)

print()
print("ASCII size:")
print(
    f"{ascii_width} columns x {ascii_height} rows"
)

print()
print("Output image size:")
print(
    f"{ascii_image.width} x {ascii_image.height}"
)

print()
print("Your ASCII portrait has been generated successfully.")