from pathlib import Path
import html

import cv2
import numpy as np
from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "assets" / "portrait.jpg"

OUTPUT_DIR = ROOT / "generated"
OUTPUT = OUTPUT_DIR / "ascii.svg"

MASK_DEBUG = OUTPUT_DIR / "foreground_mask.png"


# ============================================================
# ASCII SETTINGS
# ============================================================

COLS = 110

# Dark -> bright.
# Space means "don't draw".
ASCII_CHARS = " .,:;irsXA253hMHGS#9B&@"


# ============================================================
# SVG SETTINGS
# ============================================================

SVG_WIDTH = 900
SVG_HEIGHT = 900

BACKGROUND = "#111827"
ASCII_COLOR = "#E5E5E5"

# Portrait area inside SVG
PORTRAIT_X = 95
PORTRAIT_Y = 65
PORTRAIT_W = 710
PORTRAIT_H = 760


# ============================================================
# ANIMATION SETTINGS
# ============================================================

# Character-by-character movement:
#
# LEFT -> RIGHT
# then next row
# then LEFT -> RIGHT
#
CHAR_DELAY = 0.0001

# Small pause between rows.
ROW_DELAY = 0.002

# Individual character fade-in.
FADE_DURATION = 0.02
CHAR_STEP = 0.003

# ============================================================
# BACKGROUND REMOVAL
# ============================================================

def remove_background(image):
    """
    Use OpenCV GrabCut to separate the centered person
    from the surrounding background.
    """

    rgb = np.array(image.convert("RGB"))

    # OpenCV expects BGR.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]

    # Start with a central rectangle.
    #
    # The assumption here matches your portrait:
    # person is near the center and background surrounds them.
    margin_x = int(w * 0.01)
    margin_y = int(h * 0.02)

    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y
    )

    # GrabCut mask.
    mask = np.zeros(
        (h, w),
        np.uint8
    )

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        bgr,
        mask,
        rect,
        bgd_model,
        fgd_model,
        6,
        cv2.GC_INIT_WITH_RECT
    )

    # GrabCut labels:
    #
    # 0 = sure background
    # 2 = probable background
    # 1 = sure foreground
    # 3 = probable foreground
    #
    foreground = np.where(
        (mask == cv2.GC_FGD)
        | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)
    
    
    # Clean tiny holes/noise.
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Keep the useful central connected components.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8
    )

    cleaned = np.zeros_like(foreground)

    image_center_x = w / 2
    image_center_y = h / 2

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area < (w * h) * 0.002:
            continue

        cx, cy = centroids[label]

        # Prefer components around the center.
        distance = np.sqrt(
            ((cx - image_center_x) / w) ** 2
            + ((cy - image_center_y) / h) ** 2
        )

        if distance < 0.50:
            cleaned[labels == label] = 255

    # If the connected-component cleanup became too strict,
    # fall back to the original GrabCut mask.
    if cv2.countNonZero(cleaned) < (w * h) * 0.02:
        cleaned = foreground

    # Slight blur makes the boundary smoother.
    cleaned = cv2.GaussianBlur(
        cleaned,
        (5, 5),
        0
    )

    # Save debug mask.
    cv2.imwrite(
        str(MASK_DEBUG),
        cleaned
    )

    return cleaned


# ============================================================
# CROP SUBJECT
# ============================================================

def crop_to_subject(image, mask):

    ys, xs = np.where(mask > 60)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(
            "Background removal failed: no foreground found."
        )

    x1 = int(xs.min())
    x2 = int(xs.max())

    y1 = int(ys.min())
    y2 = int(ys.max())

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    # Add a little padding.
    pad_x = int(width * 0.19)
    pad_y = int(height * 0.09)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(image.width - 1, x2 + pad_x)

# Stop portrait around the collar.
    y2 = min(
    image.height - 1,
    y1 + int(height * 0.75)
    )
    
    cropped_image = image.crop(
        (x1, y1, x2 + 1, y2 + 1)
    )

    cropped_mask = Image.fromarray(
        mask
    ).crop(
        (x1, y1, x2 + 1, y2 + 1)
    )

    return (
        cropped_image,
        cropped_mask
    )


# ============================================================
# IMAGE -> ASCII
# ============================================================

def image_to_ascii(image, mask):

    # Orientation.
    image = ImageOps.exif_transpose(image)

    # Grayscale.
    gray = image.convert("L")

    # Mask.
    mask = mask.convert("L")

    # --------------------------------------------------------
    # Aspect ratio.
    #
    # ASCII characters are taller than they are wide,
    # therefore 0.48 keeps the portrait visually correct.
    # --------------------------------------------------------

    width, height = gray.size

    aspect = height / width

    rows = max(
        1,
        int(COLS * aspect * 0.48)
    )

    gray = gray.resize(
        (COLS, rows),
        Image.Resampling.LANCZOS
    )

    mask = mask.resize(
        (COLS, rows),
        Image.Resampling.LANCZOS
    )

    gray_array = np.asarray(gray)
    mask_array = np.asarray(mask)

    result = []

    for y in range(rows):

        line = []

        for x in range(COLS):

            # ------------------------------------------------
            # Background test.
            #
            # If this cell isn't sufficiently part of the
            # person, leave it completely blank.
            # ------------------------------------------------

            mask_value = mask_array[y, x]

            if mask_value < 70:
                line.append(" ")
                continue

            brightness = gray_array[y, x]

            # Bright subject areas use dense characters.
            index = int(
                brightness
                / 255
                * (len(ASCII_CHARS) - 1)
            )

            char = ASCII_CHARS[index]

            # If the character happens to be a space,
            # don't draw it.
            line.append(char)

        result.append(
            "".join(line).rstrip()
        )

    # Remove completely empty rows at beginning/end.
    while result and not result[0].strip():
        result.pop(0)

    while result and not result[-1].strip():
        result.pop()

    return result


# ============================================================
# SVG ANIMATION
# ============================================================

def build_svg(ascii_lines):

    svg = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    svg.append(
        f'''<?xml version="1.0" encoding="UTF-8"?>

<svg
    xmlns="http://www.w3.org/2000/svg"
    width="{SVG_WIDTH}"
    height="{SVG_HEIGHT}"
    viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">

    <rect
        x="0"
        y="0"
        width="{SVG_WIDTH}"
        height="{SVG_HEIGHT}"
        fill="{BACKGROUND}"/>

    <g
        font-family="Consolas, 'Courier New', monospace"
        font-size="9px"
        font-weight="400"
        fill="{ASCII_COLOR}">
'''
    )

    rows = len(ascii_lines)

    # Character spacing.
    char_width = PORTRAIT_W / COLS

    # Make total portrait height fit the SVG.
    line_height = min(
        10.0,
        PORTRAIT_H / max(rows, 1)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Each row is completed LEFT -> RIGHT.
    # THEN next row starts.
    #
    # This is the reference-style sequence.
    # --------------------------------------------------------
    character_counter = 0
    for row_index, row in enumerate(ascii_lines):

        y = (
            PORTRAIT_Y
            + row_index * line_height
        )

        # First character in this row starts after
        # all previous rows have completed.

        for col_index, char in enumerate(row):

            if char == " ":
                continue

            x = (
                PORTRAIT_X
                + col_index * char_width
            )

            delay = character_counter * CHAR_STEP  

            character_counter += 1

            safe_char = html.escape(
                char,
                quote=False
            )


            svg.append(
                f'''
        <text
            x="{x:.2f}"
            y="{y:.2f}"
            opacity="0"
        >{safe_char}<animate
                attributeName="opacity"
                from="0"
                to="1"
                begin="{delay:.3f}s"
                dur="{FADE_DURATION:.3f}s"
                fill="freeze"/></text>
'''
            )

    svg.append(
        '''
    </g>
</svg>
'''
    )

    return "".join(svg)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Portrait not found:\n{INPUT}"
        )

    print()
    print("==============================================")
    print("  ASCII PORTRAIT GENERATOR")
    print("==============================================")
    print()

    print("Loading portrait...")

    image = Image.open(INPUT)

    print(
        f"Original size: "
        f"{image.width} x {image.height}"
    )

    print("Removing background...")

    mask_array = remove_background(
        image
    )

    print("Cropping to subject...")

    cropped_image, cropped_mask = crop_to_subject(
        image,
        mask_array
    )

    print(
        f"Subject crop: "
        f"{cropped_image.width} x {cropped_image.height}"
    )

    print("Converting subject to ASCII...")

    ascii_lines = image_to_ascii(
        cropped_image,
        cropped_mask
    )

    print(
        f"ASCII grid: "
        f"{COLS} x {len(ascii_lines)}"
    )

    print("Building character animation...")

    svg = build_svg(
        ascii_lines
    )

    OUTPUT.write_text(
        svg,
        encoding="utf-8"
    )

    total_rows = len(ascii_lines)

    estimated_time = (
        total_rows
        * (
            COLS * CHAR_DELAY
            + ROW_DELAY
        )
        + FADE_DURATION
    )

    print()
    print("==============================================")
    print("DONE")
    print("==============================================")
    print()
    print(f"SVG output : {OUTPUT}")
    print(f"Mask debug : {MASK_DEBUG}")
    print()
    print(f"Background : {BACKGROUND}")
    print(f"ASCII      : {ASCII_COLOR}")
    print()
    print("Animation  : TOP -> BOTTOM")
    print("Each row   : LEFT -> RIGHT")
    print("Method     : CHARACTER BY CHARACTER")
    print(
        f"Time       : ~{estimated_time:.1f} seconds"
    )
    print()


if __name__ == "__main__":
    main()