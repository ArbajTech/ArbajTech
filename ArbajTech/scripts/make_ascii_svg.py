from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_IMAGE = ROOT / "assets" / "portrait.jpg"
OUTPUT_DIR = ROOT / "generated"
OUTPUT_SVG = OUTPUT_DIR / "ascii.svg"

COLUMNS = 100

# ASCII characters from dark -> light
ASCII_CHARS = "@%#*+=-:. "


# ============================================================
# IMAGE -> ASCII
# ============================================================

def image_to_ascii(image_path: Path, columns: int):
    image = Image.open(image_path).convert("RGB")

    # Keep portrait proportions while compensating for
    # terminal/monospace character height.
    width, height = image.size

    aspect_ratio = height / width
    rows = max(1, int(columns * aspect_ratio * 0.48))

    image = image.resize((columns, rows))

    # Mild processing for clearer facial details.
    image = image.filter(
        ImageFilter.GaussianBlur(radius=0.25)
    )

    gray = image.convert("L")

    gray = ImageEnhance.Contrast(gray).enhance(1.35)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)

    pixels = list(gray.getdata())

    lines = []

    for y in range(rows):
        line = []

        for x in range(columns):
            value = pixels[y * columns + x]

            index = int(
                value / 255 * (len(ASCII_CHARS) - 1)
            )

            line.append(ASCII_CHARS[index])

        lines.append("".join(line).rstrip())

    return lines, columns, rows


# ============================================================
# ASCII -> ANIMATED SVG
# ============================================================

def create_svg(lines, columns, rows):
    char_width = 9
    line_height = 12

    width = columns * char_width
    height = rows * line_height + 30

    background = "#050505"
    foreground = "#00ff9c"

    svg = []

    svg.append(
        f'''<svg xmlns="http://www.w3.org/2000/svg"
        width="{width}"
        height="{height}"
        viewBox="0 0 {width} {height}">
'''
    )

    svg.append(
        f'<rect width="100%" height="100%" fill="{background}"/>'
    )

    svg.append(
        f'''
<style>
.ascii {{
    font-family: "Cascadia Mono", "Consolas", "Courier New", monospace;
    font-size: 11px;
    font-weight: 600;
    fill: {foreground};
    letter-spacing: 0px;
}}
.cursor {{
    fill: {foreground};
}}
</style>
'''
    )

    # One clipping rectangle per row.
    # The rectangle grows from left to right,
    # creating a typewriter/reveal effect.
    for row_index, line in enumerate(lines):

        y = 18 + row_index * line_height

        safe_line = escape(line)

        clip_id = f"clip_{row_index}"

        svg.append(
            f'''
<clipPath id="{clip_id}">
    <rect x="0" y="{y - 11}"
          width="0"
          height="{line_height + 4}">
        <animate
            attributeName="width"
            from="0"
            to="{width}"
            dur="1.2s"
            begin="{row_index * 0.025:.3f}s"
            fill="freeze"/>
    </rect>
</clipPath>
'''
        )

        svg.append(
            f'''
<text
    x="0"
    y="{y}"
    class="ascii"
    clip-path="url(#{clip_id})">{safe_line}</text>
'''
        )

    # Cursor
    cursor_x = width - 8
    cursor_y = 10

    svg.append(
        f'''
<rect
    x="{cursor_x}"
    y="{cursor_y}"
    width="5"
    height="12"
    class="cursor">
    <animate
        attributeName="opacity"
        values="1;0;1"
        dur="0.9s"
        repeatCount="indefinite"/>
</rect>
'''
    )

    svg.append("</svg>")

    return "\n".join(svg)


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Portrait not found: {INPUT_IMAGE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines, columns, rows = image_to_ascii(
        INPUT_IMAGE,
        COLUMNS
    )

    svg = create_svg(
        lines,
        columns,
        rows
    )

    OUTPUT_SVG.write_text(
        svg,
        encoding="utf-8"
    )

    print("DONE")
    print(f"Input : {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_SVG}")
    print(f"ASCII : {columns} columns x {rows} rows")


if __name__ == "__main__":
    main()