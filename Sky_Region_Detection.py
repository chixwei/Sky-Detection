# import libaries
import os
import cv2
import numpy as np
import time


# Function to detect edges in an image using Canny edge detection
def detect_edges(image):
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with low threshold (40) and high threshold (150)
    edges = cv2.Canny(gray_image, 40, 150)

    return edges


# Function to connect edges to fill small holes to improve edge connectivity
def connect_edges(edges):

    # Perform morphological dilation using a smaller kernel (3x3) and 2 iterations to further connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Perform morphological closing using a larger kernel size (9x9) to fill small gaps between edges
    kernel = np.ones((9, 9), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges


# Function to detect the sky region in an image based on the connected edges
def detect_sky_region(edges_connected):
    
    # Find contours in the binary image, edges_connected is the binary image)
    # Sky region is black in edges connnected, therefore minus 255 to invert to white
    contours, _ = cv2.findContours(255 - edges_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assume that sky region is large connected area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a binary mask for the sky region uing the largest contour
    sky_mask = np.zeros_like(edges_connected)
    # Draw the largest contour (sky region) on the sky_mask with white color (255) and fill the contour (-1).
    cv2.drawContours(sky_mask, [largest_contour], 0, 255, -1)

    return sky_mask


# Function to extract the skyline of the sky region
def skyline(sky_mask):
    
    # Find contours in the sky mask
    contours, _ = cv2.findContours(sky_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for the skyline
    skyline_mask = np.zeros_like(sky_mask)

    # Draw all contours (skyline) on the skyline_mask with white color (255) and a thickness of 1.
    cv2.drawContours(skyline_mask, contours, -1, 255, 1)

    return skyline_mask


# Function to calculate accuracy between two binary masks
# pred_mask = predicted mask
# gt_mask = ground truth mask
def calculate_accuracy(pred_mask, gt_mask):
    
    # Convert the binary mask to 0 and 1 for accurate calculation
    pred_mask = np.where(pred_mask > 0, 1, 0)
    gt_mask = np.where(gt_mask > 0, 1, 0)
    
    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP = np.sum((pred_mask == 1) & (gt_mask == 1))
    TN = np.sum((pred_mask == 0) & (gt_mask == 0))
    FP = np.sum((pred_mask == 1) & (gt_mask == 0))
    FN = np.sum((pred_mask == 0) & (gt_mask == 1))

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Count as percentage by multiplying 100
    return accuracy * 100


# Function to determine whether the image is daytime or nighttime
def day_or_night(image, dataset_number):
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Count the average intensity of the grayscale image
    average_intensity = np.mean(gray_image)

    # Define a threshold to classify day or night
    day_night_threshold = 80
    
    # For dataset_number '9730', additional condition to adjust the threshold based on overexposed/saturated regions
    if dataset_number == '9730':
        
        # Check if there are overexposed/saturated regions in the grayscale image
        _, binary_image = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
        
        # Count the number of saturated pixels (white pixels) in the binary image
        num_saturated_pixels = np.sum(binary_image == 255)
    
        # If there are many saturated pixels, it's likely a night image with a bright light source
        # If the number of saturated pixels are more than 5% of the total number of pixels in the grayscale image
        if num_saturated_pixels > (gray_image.size * 0.05):
            
            # Adjust the threshold to consider it as night
            day_night_threshold = 150
            
    # Check if the average intensity of the image is below the threshold to determine day or night
    if average_intensity < day_night_threshold:
        return "Night"
    else:
        return "Day"


# Function to apply the daytime mask on the nighttime image
def apply_daytime_mask(night_image, day_mask):
    
    # Convert the nighttime image to grayscale
    gray_night_image = cv2.cvtColor(night_image, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale nighttime image using a threshold
    _, binary_night_image = cv2.threshold(gray_night_image, 0, 255, cv2.THRESH_BINARY)

    # Apply the day mask on the binary nighttime image
    # Set the non-sky regions/ ground to black
    night_sky_region = cv2.bitwise_and(binary_night_image, day_mask)

    return night_sky_region


if __name__ == '__main__':
    
    # Record the start time
    start_time = time.time()  
    
    # Store the dataset numbers into a list to loop through it
    dataset_numbers = ["623", "684", "9730", "10917"]

    # Loop through the dataset numbers in the list
    for dataset_number in dataset_numbers:
        
        # Get the current working directory where the script is located
        current_directory = os.getcwd()
        
        # Create a directory path for binarymask with the current process dataset number 
        binarymask_directory = os.path.join(current_directory, f'binarymask/{dataset_number}')
        
        # Create a directory path for masking with the current process dataset number
        masking_directory  = os.path.join(current_directory, f'masking/{dataset_number}')
        
        # Create a directory path for skyline with the current process dataset number
        skyline_directory  = os.path.join(current_directory, f'skyline/{dataset_number}')
        
        # Create the binarymask directory if it doesn't exist
        if not os.path.exists(binarymask_directory):
            os.makedirs(binarymask_directory)

        # Create the masking directory if it doesn't exist
        if not os.path.exists(masking_directory):
            os.makedirs(masking_directory)

        # Create the skyline directory if it doesn't exist
        if not os.path.exists(skyline_directory):
            os.makedirs(skyline_directory)

        # Read the ground truth mask from the file directory according to each dataset number
        ground_truth_mask = cv2.imread(f"{dataset_number}_mask.png", 0)

        # Stage 1: Process all daytime images to find the best daytime mask
        # Initialize the best day mask and best accurary to store the updated best accuracy mask and value
        best_day_mask = None
        best_accuracy = 0.0
        # List to store all the accuracy values
        accuracies = []  
        
        # Process all images from the directory
        image_directory = f'dataset/{dataset_number}/'

        # Display console output
        print(f"Data {dataset_number} is currently processing...")
        
        # Loop through all the images in the image directory (every dataset)
        for filename in os.listdir(image_directory):
            # Create a full path to the image
            image_path = os.path.join(image_directory, filename)
            # Read the current image
            image = cv2.imread(image_path)

            # First determine if it's day or night for the current image before proceed to the next step
            result = day_or_night(image, dataset_number)
            
            # Condition 1: If the image detected is day, then proceed to the following steps
            if result == "Day":

                # Detect edges in the image
                edges = detect_edges(image)

                # Connect edges with morphological operations to fill small gaps and improve connectivity
                edges_connected = connect_edges(edges)

                # Detect sky region (largest connected black region) in the edge connected image
                day_sky_mask = detect_sky_region(edges_connected)
                
                # Extract the skyline (skyline as white, others as black)
                skyline_mask = skyline(day_sky_mask)

                # Calculate accuracy of the sky region mask compared to the ground truth mask
                accuracy = calculate_accuracy(day_sky_mask, ground_truth_mask)
                # Store the accuracy value to the accuracies list
                accuracies.append(accuracy)  

                # If the current accuracy is better than the previous best accuracy
                if accuracy > best_accuracy:
                    # Save the best daytime mask and its accuracy to the initialized variables
                    best_accuracy = accuracy
                    best_day_mask = day_sky_mask

                # Save the day sky mask image
                cv2.imwrite(f"binarymask/{dataset_number}/{filename}_{accuracy:.2f}%_day.jpg", day_sky_mask)
                
                # Save the day skyline mask image
                cv2.imwrite(f"skyline/{dataset_number}/{filename}_day.jpg", skyline_mask)
                
                # Apply the mask on the original image for visualization
                sky_visualized = cv2.bitwise_and(image, image, mask=day_sky_mask)
                # Save the making image 
                cv2.imwrite(f"masking/{dataset_number}/{filename}_{accuracy:.2f}%_day.jpg", sky_visualized)

        # Stage 2: Process all nighttime images using the best daytime mask
        for filename in os.listdir(image_directory):
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)

            # Determine if it's day or night for the current image
            result = day_or_night(image, dataset_number)

            # Condition 2: If the image detected is night, then proceed to the following steps
            if result == "Night":

                # Apply the best daytime mask on the nighttime image to extract the sky region
                night_sky_region = apply_daytime_mask(image, best_day_mask)
                
                # Extract the skyline (skyline as white, others as black)
                night_skyline_mask = skyline(night_sky_region)
                
                # Calculate accuracy of the night sky region mask compared to the ground truth mask
                accuracy = calculate_accuracy(night_sky_region, ground_truth_mask)
                # Store the accuracy value to the accuracies list
                accuracies.append(accuracy)  
                
                # save the night sky mask image
                cv2.imwrite(f"binarymask/{dataset_number}/{filename}_{accuracy:.2f}%_night.jpg", night_sky_region)
                
                # Save the night skyline mask image
                cv2.imwrite(f"skyline/{dataset_number}/{filename}_night.jpg", night_skyline_mask)
                
                # Apply the mask on the original image for visualization
                night_sky_visualized = cv2.bitwise_and(image, image, mask=night_sky_region)
                # Save the making image 
                cv2.imwrite(f"masking_directory/{dataset_number}/{filename}_{accuracy:.2f}%_night.jpg", night_sky_visualized)

        
        # Calculate the total average accuracy by summing up all the accuracy values from the accuracies list and divide by total number of images
        average_accuracy = sum(accuracies) / len(accuracies)
        
        # Print the average accuracy of the whole dataset
        print(f"Average accuracy of the {dataset_number} dataset: {average_accuracy:.2f} %")
      
        
    # Record the end time of program execution
    end_time = time.time()  
    # Calculate the total execution time by substracting the end time and start time
    total_time = end_time - start_time
    # Convert total time to minutes
    total_time_minutes = total_time / 60  
    # Print the total time used to execute the whole program
    print(f"Total execution time: {total_time_minutes:.2f} minutes")


