from PIL import Image

img = Image.open("data/videos/balle_blanche.png")
print(img.size)
res1 = img.crop((300,400,2200,1600))
res1.save("data/videos/balle_blanche_crop.png",'png')

img = Image.open("data/videos/balle_orange_2.png")
print(img.size)
res2 = img.crop((300,400,2200,1600))
res2.save("data/videos/balle_orange_2_crop.png",'png')