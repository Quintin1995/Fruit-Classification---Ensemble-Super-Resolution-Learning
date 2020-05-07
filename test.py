# import cv2
# import matplotlib.pyplot as plt

path = r'data/1x/Banana/banaan0.png'

# gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

# cv2.imshow("gray",gray)




import cv2
  
# image = cv2.imread(path)
gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


# cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
  
cv2.waitKey(0)
cv2.destroyAllWindows()