import cv2 

#loads some pre-trained data on face frontals from open cv
trained_face_data= cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#choose an image to detect face in
# img= cv2.imread ('IMG_3332.JPG')
#getting default webcam video, args: index of the camera, 0 is the default cam, or name of a video file
cap = cv2.VideoCapture('Alison.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('SmileDetector.avi', fourcc, 20.0, size)

#loop forever in frames
while True:
    #read the current frame, returns true or false and the frame
    successful_frame_read, frame=cap.read()



    #convert images to black and white, args: src image and conversion of the color
    grayscaled_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img, scaleFactor=1.1, minNeighbors=10)

    #scaleFactor: how much you wanna blur the image to find the brightness>highest number, more blur
    #minNeighbore: minimum amount of neighbor rectangles>higher number, more rectangles
    
    for (x, y, w, h) in face_coordinates:
        #draw the rectangle on the frame
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,200,50), 2 )

        #making a dub image of the face from the current frame

        #slice the face using numpy N-dimensional array slicing
        the_face= frame[y:y+h, x:x+w]

        face_greyscale= cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles=smile_detector.detectMultiScale(face_greyscale)
        
        #smile loop in the face sub image
        # for (xs, ys, ws, hs) in smiles:
        #     #draw the rectangle on the frame
        #     cv2.rectangle(the_face, (xs,ys), (xs+ws, ys+hs), (50,50,200), 2 )


        #Label this face as smiling
        if len(smiles)>0 :
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255,255,255))


    # cv2.imshow('Window name', grayscaled_img)
    cv2.imshow('Recording...', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release the video capture object
cap.release()
out.release()
cv2.destroyAllWindows()

# Draw rectangles, args:  src img, coordinates, RGB color of the rectangle, the thickness of the rectangle



# print(face_coordinates)



# cv2.imshow('Clever programer', grayscaled_img)
# paueses until a key is pressed
# cv2.waitKey()



# print("Code ran completely")