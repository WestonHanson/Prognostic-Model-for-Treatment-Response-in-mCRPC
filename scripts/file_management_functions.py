# Author: Weston Hanson
# Place: Fred Hutch Cancer Center, Seattle, WA
# Date Created: 10/20/2025
# Purpose: To save model and AUC curves to version directory.

import os
import sys

def model_version_directory(model_ver, model_subver):
    """
    Parameters:
    -----------
        model_subver: String
            Name of model version.
            
        model_ver: String
            Name of model subversion.

    Function:
    ---------
        - Creates a directory for model version and subverion. If directory already exists it will overwrite any old plots or models.
        
    Returns:
    --------
        version_dir: String
            File directory to save plots and models.
    """
    print("\n" + "="*80)
    print("VERSION MAINTENANCE")
    print("="*80 + "\n")

    while True:
        print(f"Do you want to save (or use) model as {model_subver} ? (y/N):")
        user_answer = input()
        if user_answer == "N":
            print("Exiting...\n")
            sys.exit(0)

        elif user_answer == "y":
            # Create version directory      
            version_dir = os.path.join("..", "saved-models", model_ver, model_subver)
            try:
                if os.path.exists(version_dir):
                    if os.path.isdir(version_dir):
                        while True:
                            print(f"\nModel {model_subver} already exists. Do you want to use existing model (meaning use it to predict on holdout set)? (y/N):")
                            use_model = input("")
                            if use_model == "y":
                                print(f"\nUsing existing model {model_subver} to predict on holdout set...\n")
                                return version_dir, "use_model"
                            print(f"\nDo you want to continue and overwrite the existing model? (y/N):")
                            overwrite = input("")
                            if overwrite == "y":
                                print(f"\nOverwritting existing model {model_subver}...\n")
                                break
                            elif overwrite == "N":
                                print("Exiting...\n")
                                sys.exit(0)
                            else:
                                print("Try again.\n")
                    else:
                        raise NotADirectoryError(f"Path exists but is not a directory: {version_dir}")
                else:
                    os.makedirs(version_dir)
                    if os.path.isdir(version_dir):
                        print(f"{version_dir} directory made.")
            except FileNotFoundError:
                print(f"Error: Parent directory for {version_dir} does not exist.")
            
            print(f"\nRunning model as {model_subver}...\n")
            break
        
        else:
            print("Try again.\n")

    return version_dir, None