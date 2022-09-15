# 크롤링 끝
from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"NCT 도영,이세영,백진희,데이식스 원필,있지 유나","limit":100,"print_urls":True,"format":"jpg"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the pdownloaded images