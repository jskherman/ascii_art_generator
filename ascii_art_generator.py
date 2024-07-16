"""
ASCII Art Generator
===================

This script converts input images into ASCII art, incorporating edge detection techniques
and offering options for color output. This is based on the [ReShade ASCII
shader by Acerola](https://www.youtube.com/watch?v=gg40RWiaHRY) for FFXIV.
Check out his shader [on GitHub](https://github.com/GarrettGunnell/AcerolaFX/blob/main/Shaders/AcerolaFX_ASCII.fx).

Detailed Process
----------------
1. Image Loading and Preprocessing:
   - The input image is loaded using PIL (Python Imaging Library) and converted to grayscale.
   - The image is resized to fit within a specified maximum width while maintaining its aspect ratio.
     This resizing accounts for the typical aspect ratio difference between characters and pixels,
     ensuring the output ASCII art is not vertically stretched.

2. Edge Detection:
   - The script employs the Sobel operator, a discrete differentiation operator, to detect edges in the image.
   - The process involves:
     a) Defining Sobel kernels for x and y directions. These kernels are 3x3 matrices used for convolution.
     b) Convolving the image with these kernels to compute gradients in x and y directions.
        Convolution is a mathematical operation that slides the kernel over the image, performing element-wise
        multiplication and summing the results to produce gradient values.
     c) Calculating the gradient magnitude and direction from these x and y gradients using vector operations.
   - The gradient directions are quantized (discretized) into four categories to simplify edge representation:
     vertical, diagonal (/), horizontal, and diagonal (\\\\).

3. ASCII Character Mapping:
   - For non-edge areas, pixel intensities are mapped to ASCII characters.
     The mapping uses a string of characters ordered by increasing visual "weight" or density:
     " .,:-=+*#%@"
     This ordering creates a grayscale-like effect in the ASCII output.
   - For edge areas (determined by comparing gradient magnitude to a threshold), direction-based characters are used:
     "|" for vertical edges, "/" for diagonal (/) edges, "-" for horizontal edges, and "\" for diagonal (\\\\) edges.
     This enhances the visibility of edges in the ASCII output.

4. ASCII Art Generation:
   - The script iterates over each pixel in the resized image.
   - For each pixel, it determines whether it's part of an edge based on the gradient magnitude and threshold.
   - The appropriate character (edge-based or intensity-based) is selected and added to the ASCII output.
   - These characters are collected into lines, forming the complete ASCII art representation.

5. Output Generation:
   - The ASCII art is written to a plain text file, which can be viewed in any text editor.
   - Optionally, a color output can be generated in PNG or SVG format:
     a) For PNG:
        - A new image is created with a dark background using PIL.
        - The ASCII characters are drawn on this image using a monospace font, simulating terminal output.
        - The result is saved as a PNG file, providing a raster image of the ASCII art.
     b) For SVG:
        - An SVG file is created with a dark background using XML formatting.
        - The ASCII characters are added as text elements, styled with a monospace font.
        - The result is saved as an SVG file, offering a scalable vector representation of the ASCII art.

6. Command-line Interface:
   - The script utilizes Python's argparse module to provide a user-friendly command-line interface.
   - Users can specify various parameters including input image, output text file, maximum width,
     edge detection threshold, and optional color output path.

Key Components
--------------
- Gaussian and Sobel filters: Used for edge detection, these are fundamental image processing techniques.
- PIL (Python Imaging Library): A powerful library for opening, manipulating, and saving various image file formats.
- NumPy: A library for efficient array operations, crucial for handling image data as multi-dimensional arrays.
- SVG generation: Allows for creating scalable vector graphics, useful for high-quality, resolution-independent output.

This script demonstrates the application of image processing techniques to create detailed,
edge-aware ASCII representations of input images. It offers flexibility in output formats,
catering to different use cases from simple text-based art to more sophisticated vector graphics.

Usage
-----

`python ascii_art_generator.py input_image output_text [--max_width MAX_WIDTH] [--edge_threshold EDGE_THRESHOLD] [--color_output COLOR_OUTPUT.(png|svg)] [--color_overlay COLOR]`
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import convolve
import math


def gaussian(sigma, pos):
    """
    Calculate the Gaussian function value for a given sigma and position.

    The Gaussian function, also known as the normal distribution, is a bell-shaped curve
    that is symmetric about the mean. It's widely used in image processing for smoothing
    and noise reduction.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distribution. Controls the "spread" of the distribution.
    pos : float
        Position at which to calculate the Gaussian value, typically the distance from the center.

    Returns
    -------
    float
        Gaussian function value at the given position.
    """
    return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * math.exp(
        -(pos * pos) / (2.0 * sigma * sigma)
    )


def create_gaussian_kernel(kernel_size, sigma):
    """
    Create a 2D Gaussian kernel for image convolution.

    A Gaussian kernel is used in image processing for blurring and noise reduction.
    It's a matrix of values that follow a Gaussian distribution, with higher values
    near the center and lower values towards the edges.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel (must be odd). Larger sizes create more blur but are computationally expensive.
    sigma : float
        Standard deviation of the Gaussian distribution. Controls the amount of blurring.

    Returns
    -------
    numpy.ndarray
        2D Gaussian kernel normalized so that its elements sum to 1.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = gaussian(sigma, math.sqrt(x * x + y * y))
    return kernel / np.sum(kernel)  # Normalize the kernel


def sobel_filter(image):
    """
    Apply Sobel filter to detect edges in the image.

    The Sobel filter is an edge detection operator that computes an approximation of the
    gradient of the image intensity function. It emphasizes regions of high spatial frequency
    that correspond to edges.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a 2D numpy array.

    Returns
    -------
    tuple
        Gradient magnitude and gradient direction arrays.
    """
    # Define Sobel kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # Compute gradients in x and y directions using convolution
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    # Compute gradient magnitude and direction
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_direction = np.arctan2(grad_y, grad_x)

    return grad_magnitude, grad_direction


def quantize_directions(directions):
    """
    Quantize gradient directions into 4 categories for simplified edge representation.

    This function discretizes the continuous gradient directions into four main categories,
    which helps in assigning appropriate ASCII characters for edge representation.

    Parameters
    ----------
    directions : numpy.ndarray
        Array of gradient directions in radians.

    Returns
    -------
    numpy.ndarray
        Quantized directions (0: vertical, 1: diagonal /, 2: horizontal, 3: diagonal \\\\).
    """
    quantized = np.zeros_like(directions, dtype=int)
    angles = np.abs(directions) % (2 * np.pi)  # Wrap angles within the range [0, 2π]

    # Convert angles greater than π to their negative equivalents
    angles[angles > np.pi] -= 2 * np.pi

    # Quantize angles into 4 categories based on their value in radians
    quantized[
        (angles < np.pi / 8) | (angles >= 7 * np.pi / 8) | (angles < -7 * np.pi / 8)
    ] = 0  # Vertical
    quantized[
        (angles >= np.pi / 8) & (angles < 3 * np.pi / 8)
        | (angles <= -5 * np.pi / 8) & (angles > -7 * np.pi / 8)
    ] = 1  # Diagonal (/)
    quantized[
        (angles >= 3 * np.pi / 8) & (angles < 5 * np.pi / 8)
        | (angles <= -3 * np.pi / 8) & (angles > -5 * np.pi / 8)
    ] = 2  # Horizontal
    quantized[
        (angles >= 5 * np.pi / 8) & (angles < 7 * np.pi / 8)
        | (angles <= -np.pi / 8) & (angles > -3 * np.pi / 8)
    ] = 3  # Diagonal (\)

    return quantized


def get_ascii_char(intensity):
    """
    Map pixel intensity to an ASCII character.

    This function creates a visual representation of different pixel intensities
    using ASCII characters of varying "weight" or visual density.

    Parameters
    ----------
    intensity : float
        Pixel intensity value (0-255).

    Returns
    -------
    str
        Corresponding ASCII character.
    """
    # ascii_chars = " .,:=+*#%@"
    ascii_chars = " .:coP0?@■"
    # ascii_chars = "⠀⠁⠉⠛⠿⣿█"

    index = int(intensity / 256 * len(ascii_chars))
    return ascii_chars[min(index, len(ascii_chars) - 1)]


def resize_image(image, max_width):
    """
    Resize the image to fit within the specified maximum width while maintaining aspect ratio.

    This function ensures that the output ASCII art has appropriate dimensions,
    taking into account the difference in aspect ratio between characters and pixels.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    max_width : int
        Maximum width for the resized image in characters.

    Returns
    -------
    PIL.Image
        Resized image.
    """
    width, height = image.size
    aspect_ratio = height / width
    new_width = min(max_width, width)
    new_height = int(
        new_width * aspect_ratio * 0.55
    )  # 0.55 compensates for terminal line spacing
    return image.resize((new_width, new_height), Image.LANCZOS)


def image_to_ascii(
    image_path,
    output_path,
    max_width=100,
    edge_threshold=30,
    color_output=None,
    color_overlay="#F2613F",
):
    """
    Convert an image to ASCII art and optionally create a color output.

    This function orchestrates the entire process of converting an image to ASCII art,
    including edge detection, character mapping, and optional color output generation.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str
        Path to save the ASCII art text file.
    max_width : int, optional
        Maximum width of the ASCII art in characters (default is 100).
    edge_threshold : int, optional
        Threshold for edge detection (default is 30). Higher values result in fewer detected edges.
    color_output : str, optional
        Path to save color output (PNG or SVG).

    Returns
    -------
    None
    """
    # Load image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Resize image to fit within max_width while maintaining aspect ratio
    image = resize_image(image, max_width)

    # Convert image to numpy array for efficient processing
    img_array = np.array(image, dtype=np.float32)

    # Apply Sobel filter for gradient magnitude and direction
    grad_magnitude, grad_direction = sobel_filter(img_array)

    # Normalize gradient magnitude to 0-255 range
    grad_magnitude = (grad_magnitude / np.max(grad_magnitude)) * 255

    # Quantize directions for simplified edge representation
    quantized_directions = quantize_directions(grad_direction)

    # Generate ASCII art
    ascii_art = []
    height, width = img_array.shape
    for y in range(height):
        line = []
        for x in range(width):
            if grad_magnitude[y, x] > edge_threshold:
                # Use direction-based characters for edges
                direction = quantized_directions[y, x]
                if direction == 0:
                    line.append("|")
                elif direction == 1:
                    line.append("/")
                elif direction == 2:
                    line.append("—")
                elif direction == 3:
                    line.append("\\")
            else:
                # Use intensity-based characters for non-edge areas
                intensity = img_array[y, x]
                line.append(get_ascii_char(intensity))
        ascii_art.append("".join(line))

    # Write ASCII art to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ascii_art))

    # Generate color output if requested
    if color_output:
        if color_output.lower().endswith(".png"):
            create_color_png(ascii_art, color_output, color_overlay)
        elif color_output.lower().endswith(".svg"):
            create_color_svg(ascii_art, color_output, color_overlay)
        else:
            print("Error: Unsupported colored output format. Use PNG or SVG.")


def create_color_png(ascii_art, output_path, color_overlay="#F2613F"):
    """
    Create a color PNG image of the ASCII art.

    This function generates a raster image representation of the ASCII art,
    simulating the appearance of colored text on a dark terminal.

    Parameters
    ----------
    ascii_art : list
        List of strings representing ASCII art lines.
    output_path : str
        Path to save the output PNG file.

    Returns
    -------
    None
    """
    font_size = 12
    try:
        # Attempt to use a custom monospace font
        font = ImageFont.truetype("CaskaydiaMonoNerdFontMono.ttf", font_size)
    except IOError:
        # Fall back to default font if custom font is not available
        font = ImageFont.load_default()

    # Calculate image dimensions based on font metrics
    sample_text = "A"
    char_width, char_height = font.getbbox(sample_text)[
        2:4
    ]  # Width and height of a character
    img_width = max(len(line) for line in ascii_art) * char_width
    img_height = len(ascii_art) * char_height

    # Create image with a dark background
    image = Image.new("RGB", (img_width, img_height), "#0C0C0C")
    draw = ImageDraw.Draw(image)

    # Draw ASCII art using the chosen font and color
    for y, line in enumerate(ascii_art):
        draw.text((0, y * char_height), line, font=font, fill=color_overlay)

    # Save image in PNG format
    image.save(output_path)


def create_color_svg(ascii_art, output_path, color_overlay="#F2613F"):
    """
    Create a color SVG image of the ASCII art.

    This function generates a vector graphics representation of the ASCII art,
    allowing for scalable and high-quality output.

    Parameters
    ----------
    ascii_art : list
        List of strings representing ASCII art lines.
    output_path : str
        Path to save the output SVG file.

    Returns
    -------
    None
    """
    font_size = 12
    char_width, char_height = 8, 16  # Approximate dimensions for a monospace font

    # Calculate image dimensions
    img_width = (
        max(len(line) for line in ascii_art) * char_width * 0.88
    )  # Adjusted width to remove extra space
    img_height = len(ascii_art) * char_height

    # Prepare SVG content with styling
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{img_width}" height="{img_height}">
<style>
    text {{
        font-family: "CaskaydiaMono Nerd Font", "Cascadia Code", monospace;
        font-size: {font_size}px;
        fill: {color_overlay};
    }}
</style>
<rect width="100%" height="100%" fill="#0C0C0C"/>
"""

    # Draw ASCII art as text elements in the SVG
    for y, line in enumerate(ascii_art):
        y_pos = (y + 1) * char_height - 2  # Adjust vertical position
        # Escape special characters to ensure valid XML
        escaped_line = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ", " ")  # Replace space with non-breaking space
        )
        svg_content += f'<text x="0" y="{y_pos}">{escaped_line}</text>\n'

    svg_content += "</svg>"

    # Write SVG content to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)


if __name__ == "__main__":
    """
    Main entry point of the script.
    
    This section sets up the command-line interface using argparse,
    allowing users to specify input parameters when running the script.
    """
    parser = argparse.ArgumentParser(
        description="Convert an image to ASCII art with improved edge detection and optional color output."
    )
    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument("output_text", type=str, help="Path to the output text file")
    parser.add_argument(
        "--max_width",
        type=int,
        default=100,
        help="Maximum width of the ASCII art in characters (default: 100)",
    )
    parser.add_argument(
        "--edge_threshold",
        type=int,
        default=30,
        help="Threshold for edge detection (default: 30)",
    )
    parser.add_argument(
        "--color_output",
        type=str,
        help="Path to save color output (PNG or SVG)",
    )
    parser.add_argument(
        "--color-overlay",
        type=str,
        help="Choose the color overlay for the ASCII art",
        default="#F2613F",
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    image_to_ascii(
        args.input_image,
        args.output_text,
        max_width=args.max_width,
        edge_threshold=args.edge_threshold,
        color_output=args.color_output,
        color_overlay=args.color_overlay,
    )
