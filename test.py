ZIP = '/home/ubuntu/user/jihye.lee/data/detection_aihub/box_data.zip'
import zipfile
import io
archive = zipfile.ZipFile(ZIP, 'r')
files = archive.namelist()
data = archive.read(files[0])
data = data.decode('utf-8')
# print(data.decode('utf8'))
print(data)