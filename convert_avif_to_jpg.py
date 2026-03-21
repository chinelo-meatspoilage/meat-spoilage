"""
Convert all non-JPEG images in a folder to JPEG with a white background.

Handles: WEBP, AVIF, PNG, BMP, TIFF, GIF, JPEG (.jpeg rename), and any other
Pillow-readable format.  Transparent or non-white backgrounds are composited
onto solid white before encoding as JPEG so no dark/transparent patches appear.

Usage:
    python convert_avif_to_jpg.py [folder_path]

Default folder is holdout/Fresh.  The script deletes each original file only
after it has been successfully saved as .jpg.

Requires:
    pip install pillow pillow-avif-plugin
"""

import sys
from pathlib import Path
from PIL import Image

# Register AVIF support when pillow-avif-plugin is installed.
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

# Extensions that need to be converted or renamed to .jpg
NON_JPEG_EXTENSIONS = {".webp", ".avif", ".png",
                       ".bmp", ".tiff", ".tif", ".gif", ".jpeg"}


def composite_on_white(img: Image.Image) -> Image.Image:
    """Return an RGB image composited onto a solid white background."""
    # Ensure we are working with RGBA so the paste mask is always correct.
    if img.mode in ("P", "PA", "LA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        white_bg = Image.new("RGB", img.size, (255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])  # channel 3 = alpha
        return white_bg
    return img.convert("RGB")


def convert_to_jpeg(img_path: Path) -> None:
    """Open any image, composite onto white if needed, save as .jpg, delete original."""
    img = Image.open(img_path)
    rgb = composite_on_white(img)

    out_path = img_path.with_suffix(".jpg")
    # Guard against silently overwriting an existing .jpg with a different name.
    if out_path.exists() and out_path.resolve() != img_path.resolve():
        out_path = img_path.with_name(img_path.stem + "_converted.jpg")

    rgb.save(out_path, "JPEG", quality=95)
    img_path.unlink()
    print(f"  Converted: {img_path.name}  ->  {out_path.name}")


def process_folder(folder: Path) -> None:
    converted = 0
    failed = 0
    already_jpg = 0

    for img_path in sorted(folder.iterdir()):
        if not img_path.is_file():
            continue
        ext = img_path.suffix.lower()
        if ext == ".jpg":
            already_jpg += 1
            continue
        if ext not in NON_JPEG_EXTENSIONS:
            print(f"  Skipping (unknown extension): {img_path.name}")
            continue
        try:
            convert_to_jpeg(img_path)
            converted += 1
        except Exception as exc:
            print(f"  FAILED: {img_path.name}: {exc}")
            failed += 1

    print(f"\nSummary for '{folder}':")
    print(f"  Already .jpg : {already_jpg}")
    print(f"  Converted    : {converted}")
    print(f"  Failed       : {failed}")


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("holdout/Fresh")
    if not folder.is_dir():
        print(f"Error: folder not found: {folder}")
        sys.exit(1)
    print(f"Processing folder: {folder.resolve()}\n")
    process_folder(folder)


if __name__ == "__main__":
    main()
