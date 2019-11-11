from PIL import Image, ImageDraw, ImageFont
import os

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

text = "abcdefghijklmnopqrstuvwxyz"
 
for i in range(len(fonts)):
    fnt = ImageFont.truetype(fonts[i], 30)
    folder_name = fonts[i].split('/')[-1].split('.')[0]
    # os.system('mkdir ./datasets/"{}"'.format(folder_name))

    for j, letter in enumerate(text):
        img = Image.new('RGB', (24, 40), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((4, 0), letter, font=fnt, fill=(0, 0, 0))
        img.save('./datasets/{}/{}.png'.format(folder_name, letter))