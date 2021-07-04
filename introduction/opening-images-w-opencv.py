import cv2

img = cv2.imread("../data/00-puppy.jpg")

while True:
    cv2.imshow("Puppy", img)
    # TODO : Buraya bak

    if cv2.waitKey(1) & 0xFF == ord("a"):
        break

cv2.destroyAllWindows()



