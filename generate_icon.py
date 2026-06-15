from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs('assets', exist_ok=True)

# === icon.png ===
icon = Image.new('RGBA', (1024, 1024), color='#0A0A0A')
draw = ImageDraw.Draw(icon)
draw.ellipse([112, 112, 912, 912], outline='#00FFCC', width=20)
draw.arc([200, 200, 824, 824], start=200, end=340, fill='#00FFCC', width=25)
draw.ellipse([480, 480, 544, 544], fill='#00FFCC')
draw.line([(512, 512), (350, 750)], fill='#00FFCC', width=20)
draw.arc([600, 300, 750, 450], start=180, end=270, fill='#00FFCC', width=12)
draw.arc([650, 250, 800, 400], start=180, end=270, fill='#00FFCC', width=8)
draw.ellipse([492, 492, 532, 532], fill='#FFFFFF')
icon.save('assets/icon.png')
print("✅ assets/icon.png")

# === adaptive-icon.png ===
adaptive = Image.new('RGBA', (1024, 1024), color=(0,0,0,0))
draw2 = ImageDraw.Draw(adaptive)
draw2.ellipse([162, 162, 862, 862], outline='#00FFCC', width=20)
draw2.arc([250, 250, 774, 774], start=200, end=340, fill='#00FFCC', width=25)
draw2.ellipse([480, 480, 544, 544], fill='#00FFCC')
draw2.line([(512, 512), (350, 750)], fill='#00FFCC', width=20)
draw2.arc([600, 300, 750, 450], start=180, end=270, fill='#00FFCC', width=12)
draw2.ellipse([492, 492, 532, 532], fill='#FFFFFF')
adaptive.save('assets/adaptive-icon.png')
print("✅ assets/adaptive-icon.png")

# === splash-icon.png ===
splash = Image.new('RGBA', (1242, 2436), color='#0A0A0A')
draw3 = ImageDraw.Draw(splash)
offset_x = (1242 - 1024) // 2
offset_y = (2436 - 1024) // 2 - 200
draw3.ellipse([offset_x + 112, offset_y + 112, offset_x + 912, offset_y + 912], outline='#00FFCC', width=20)
draw3.arc([offset_x + 200, offset_y + 200, offset_x + 824, offset_y + 824], start=200, end=340, fill='#00FFCC', width=25)
draw3.ellipse([offset_x + 480, offset_y + 480, offset_x + 544, offset_y + 544], fill='#00FFCC')
draw3.line([(offset_x + 512, offset_y + 512), (offset_x + 350, offset_y + 750)], fill='#00FFCC', width=20)
draw3.arc([offset_x + 600, offset_y + 300, offset_x + 750, offset_y + 450], start=180, end=270, fill='#00FFCC', width=12)
draw3.ellipse([offset_x + 492, offset_y + 492, offset_x + 532, offset_y + 532], fill='#FFFFFF')
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
    text = "AirDrop AI"
    bbox = draw3.textbbox((0, 0), text, font=font)
    text_x = (1242 - (bbox[2] - bbox[0])) // 2
    draw3.text((text_x, offset_y + 950), text, fill='#00FFCC', font=font)
    font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 50)
    text2 = "Mesh Node"
    bbox2 = draw3.textbbox((0, 0), text2, font=font2)
    text2_x = (1242 - (bbox2[2] - bbox2[0])) // 2
    draw3.text((text2_x, offset_y + 1050), text2, fill='#888888', font=font2)
except:
    draw3.text((420, offset_y + 950), "AirDrop AI", fill='#00FFCC')
    draw3.text((450, offset_y + 1050), "Mesh Node", fill='#888888')
splash.save('assets/splash-icon.png')
print("✅ assets/splash-icon.png")

# === favicon.png ===
favicon = Image.new('RGBA', (64, 64), color='#0A0A0A')
draw4 = ImageDraw.Draw(favicon)
draw4.ellipse([8, 8, 56, 56], outline='#00FFCC', width=2)
draw4.arc([14, 14, 50, 50], start=200, end=340, fill='#00FFCC', width=2)
draw4.ellipse([30, 30, 34, 34], fill='#00FFCC')
favicon.save('assets/favicon.png')
print("✅ assets/favicon.png")

print("\nAll icons generated! Add to git: git add assets/ && git commit -m 'Add icons'")
