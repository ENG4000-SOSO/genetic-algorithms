from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


image_dir = Path.cwd() / 'debug'

# Load all images
frames = [Image.open(img) for img in sorted(image_dir.glob('*.png')) if 'Maximum' in img.stem]

# Create a font object (you can adjust the font size and type)
font = ImageFont.load_default(50)
font.size += 100

# Add text to each frame
for idx, frame in enumerate(frames):
    draw = ImageDraw.Draw(frame)
    text = f'Frame {idx + 1}/{len(frames)}'
    # Positioning the text at the top-right corner (adjust as needed)
    # draw.text((frame.width - 100, 10), text, font=font, fill='white')
    draw.text((10, 10), text, font=font, fill='black')

# Save as GIF
frames[0].save(
    Path.cwd() / 'graph_animation5.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
