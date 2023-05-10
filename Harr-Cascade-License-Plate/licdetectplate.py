import cv2
import pytesseract
import numpy as np


## THis is the main file
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
cascade = cv2.CascadeClassifier('/Users/shraiyashpandey/PycharmProjects/detection_license/haarcascade_russian_plate_number.xml')

states = {"AN":"Andaman and Nicobar Islands",
   "AP":"Andhra Pradesh",
   "AR":"Arunachal Pradesh",
   "AS":"Assam",
   "BR":"Bihar",
   "CG":"Chhattisgarh",
   "CH":"Chandigarh",
   "DN":"Dadra and Nagar Haveli",
   "DD":"Daman and Diu",
   "DL":"Delhi",
   "GA":"Goa",
   "GJ":"Gujarat",
   "HR":"Haryana",
   "HP":"Himachal Pradesh",
   "JK":"Jammu and Kashmir",
   "JH":"Jharkhand",
   "KA":"Karnataka",
   "KL":"Kerala",
   "LA":"Ladakh",
   "LD":"Lakshadweep",
   "MP":"Madhya Pradesh",
   "MH":"Maharashtra",
   "MN":"Manipur",
   "ML":"Meghalaya",
   "MZ":"Mizoram",
   "NL":"Nagaland",
   "OD":"Odisha",
   "PB":"Punjab",
   "PY":"Pondicherry",
   "RJ":"Rajasthan",
   "SK":"Sikkim",
   "TN":"Tamil Nadu",
   "TS":"Telangana",
   "TR":"Tripura",
   "UP":"Uttar Pradesh",
   "UK":"Uttarakhand",
   "WB":"West Bengal"}


def extract_num(img_name):
    global read
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in nplate:
       a = int(0.02*img.shape[0])
       b = int(0.025*img.shape[1])
       plate = img[y+a:y+h-a, x+b:x+w-b, :]

       #image processing
       kernel = np.ones((1,1), np.uint8)
       plate = cv2.dilate(plate, kernel, iterations=1)
       plate = cv2.erode(plate, kernel, iterations=1)
       plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
       (thresh,plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
       #cv2.imshow("Random", plate)

       read = pytesseract.image_to_string(plate)
       # print(read[0:2])

       # print(stat1)
       # print(states['MH'])
       # print(states[stat1])
       # try:
       #    print('Car belongs to', my_list[read[0:2]])
       # except:
       #    print("State is not recognized!")
       read = ''.join(e for e in read if e.isalnum())
       stat1 = read[0:2]
       # stat = read[0:2]
       # print(stat)
       #print(read)
       try:
          print('Car belongs to', states[stat1])
          print('The license plate number:', read)
       except:
          print('State not recognised!')
       cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
       cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), -1)
       cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
       cv2.imshow('Plate', plate)

    cv2.imshow("Result", img)
    cv2.imwrite('result.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 2, 35, 36, t18
# suitable at 1.1 -> t2, t18, t20, t28, t35, t36, t39
# suitable at 1.2 -> t1, t3, t33,t37

img_name = "./test_images/t36.jpg"
extract_num(img_name)