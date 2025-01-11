import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

# Initialize the camera (use 0 for the default camera, or adjust the index for other cameras)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    sys.exit()

print("Press 's' to capture an image and read the license plate. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize the frame for display
    frame = imutils.resize(frame, width=800)
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # If 's' is pressed, process the frame
    if key == ord('s'):
        image = frame.copy()
        image = imutils.resize(image, width=500)

        # Grayscale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Detect edges
        edged = cv2.Canny(gray, 170, 200)

        # Find contours
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        NumberPlateCnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  # Select the contour with 4 corners
                NumberPlateCnt = approx
                break

        if NumberPlateCnt is not None:
            # Masking the part other than the number plate
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(image, image, mask=mask)

            cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Final Image", new_image)

            # Configuration for tesseract
            config = ('-l eng --oem 1 --psm 3')

            # Run tesseract OCR on the image
            text = pytesseract.image_to_string(new_image, config=config).strip()

            # Print recognized text in the terminal
            print("Recognized License Plate Text:")
            print("=" * len(text))  # Create a simple visual separator
            print(text)
            print("=" * len(text))

            # Save data to CSV
            raw_data = {
                'date': [time.asctime(time.localtime(time.time()))],
                'v_number': [text]
            }

            df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
            df.to_csv('data.csv', mode='a', header=False, index=False)

        else:
            print("No license plate detected. Try again.")

    # If 'q' is pressed, exit the loop
    elif key == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
