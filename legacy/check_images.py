#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# check_images.py — legacy Udacity driver (kept for reference)
#
# PROGRAMMER: Andrés Iglesias
# DATE CREATED: 2023-09-05
# REVISED DATE: 2025-08-29
#
# PURPOSE:
#   Classify pet images with a pretrained CNN (convolutional neural network)
#   — AlexNet / VGG / ResNet — compare predictions to labels parsed from
#   filenames, and print summary metrics.
#   NOTE: In this repo, the actively maintained entry point is src/evaluate.py
#         (CSV output, simple CLI). This script is preserved for transparency.
#
# USAGE:
#   python check_images.py --dir <images_dir> --arch <resnet|vgg|alexnet> \
#                          --dogfile data/dognames.txt
# EXAMPLE:
#   python check_images.py --dir data/sample/pet_images --arch vgg \
#                          --dogfile data/dognames.txt
#
# INPUTS:
#   --dir     Folder of images; filenames encode the ground-truth label
#             (e.g., "Beagle_01125.jpg") after lowercasing/stripping.
#   --arch    Pretrained model to use: resnet | vgg | alexnet
#   --dogfile Path to text file with dog breeds, one per line
#
# OUTPUTS:
#   Prints a summary to the console. For CSV output, use:
#     python -m src.evaluate --images <images_dir> --model all

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    
    # TODO 1: Define get_input_args function within the file get_input_args.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg  
    check_command_line_arguments(in_arg)

    
    # TODO 2: Define get_pet_labels function within the file get_pet_labels.py
    # Once the get_pet_labels function has been defined replace 'None' 
    # in the function call with in_arg.dir  Once you have done the replacements
    # your function call should look like this: 
    #             get_pet_labels(in_arg.dir)
    # This function creates the results dictionary that contains the results, 
    # this dictionary is returned from the function call as the variable results
    results = get_pet_labels(in_arg.dir)

    # Function that checks Pet Images in the results Dictionary using results    
    check_creating_pet_image_labels(results)


    # TODO 3: Define classify_images function within the file classiy_images.py
    # Once the classify_images function has been defined replace first 'None' 
    # in the function call with in_arg.dir and replace the last 'None' in the
    # function call with in_arg.arch  Once you have done the replacements your
    # function call should look like this: 
    #             classify_images(in_arg.dir, results, in_arg.arch)
    # Creates Classifier Labels with classifier function, Compares Labels, 
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)

    # Function that checks Results Dictionary using results    
    check_classifying_images(results)    

    
    # TODO 4: Define adjust_results4_isadog function within the file adjust_results4_isadog.py
    # Once the adjust_results4_isadog function has been defined replace 'None' 
    # in the function call with in_arg.dogfile  Once you have done the 
    # replacements your function call should look like this: 
    #          adjust_results4_isadog(results, in_arg.dogfile)
    # Adjusts the results dictionary to determine if classifier correctly 
    # classified images as 'a dog' or 'not a dog'. This demonstrates if 
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment using results
    check_classifying_labels_as_dogs(results)


    # TODO 5: Define calculates_results_stats function within the file calculates_results_stats.py
    # This function creates the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages). This
    # dictionary is returned from the function call as the variable results_stats    
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    results_stats = calculates_results_stats(results)

    # Function that checks Results Statistics Dictionary using results_stats
    check_calculating_results(results, results_stats)


    # TODO 6: Define print_results function within the file print_results.py
    # Once the print_results function has been defined replace 'None' 
    # in the function call with in_arg.arch  Once you have done the 
    # replacements your function call should look like this: 
    #      print_results(results, results_stats, in_arg.arch, True, True)
    # Prints summary results, incorrect classifications of dogs (if requested)
    # and incorrectly classified breeds (if requested)
    print_results(results, results_stats, in_arg.arch, True, True)
    
    # TODO 0: Measure total program runtime by collecting end time
    end_time = time()
    
    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
