import cv2


def ask_for_tracker():
    tracker = None
    print("Welcome! What Tracker API would you like to use?")
    print("Enter 0 for BOOSTING: ")
    print("Enter 1 for MIL: ")
    print("Enter 2 for KCF: ")
    print("Enter 3 for TLD: ")
    print("Enter 4 for MEDIANFLOW: ")
    choice = input("Please select your tracker: ")

    if choice == '0':
        tracker = cv2.legacy.TrackerBoosting_create()
    if choice == '1':
        tracker = cv2.legacy.TrackerMIL_create()
    if choice == '2':
        tracker = cv2.legacy.TrackerKCF_create()
    if choice == '3':
        tracker = cv2.legacy.TrackerTLD_create()
    if choice == '4':
        tracker = cv2.legacy.TrackerMedianFlow_create()

    return tracker


ask_for_tracker()
