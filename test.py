ZIP = '/home/ubuntu/user/jihye.lee/data/detection_aihub/box_data.zip'
import zipfile
import io
archive = zipfile.ZipFile(ZIP, 'r')
img = 'image_data/00810011001.jpg'
file_name = img.replace('image', 'box').replace('jpg', 'txt')
files = archive.namelist()
# print(files[0])
data = archive.read(file_name)
data = data.decode('utf-8')
# print(data.decode('utf8'))
for line in data.split('\n'):
    print(line)
    break