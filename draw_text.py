from PIL import Image, ImageDraw, ImageFont
 
img = Image.new('RGB', (240, 340), color = (255, 255, 255))

fonts = [
    '/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf',
    '/System/Library/Fonts/Supplemental/Brush Script.ttf',
    '/System/Library/Fonts/Supplemental/Chalkboard.ttc',
    '/System/Library/Fonts/Supplemental/Chalkduster.ttf',
    '/System/Library/Fonts/Supplemental/Courier New Italic.ttf',
    '/System/Library/Fonts/MarkerFelt.ttc',
    '/System/Library/Fonts/Supplemental/Papyrus.ttc',
    '/System/Library/Fonts/Supplemental/SnellRoundhand.ttc',
]

text = "abcdefghijk"
 
for i in range(len(fonts)):

    fnt = ImageFont.truetype(fonts[i], 30)

    for j, letter in enumerate(text):
        d = ImageDraw.Draw(img)
        d.text((10 + 20*j, 7 + 40*i), letter, font=fnt, fill=(0, 0, 0))

img.save('text_fonts.png')