from credit_application import *


def menu(applicant, model, ori_uval, encode_uval):
    """
    Function to:
    - Show the available menu option
    - Choose the menu
    - Perform the task based on the choosen menu

    :param applicant: Dictionary containing the applicant details
    :param model: andom Forest model which has been trained before
    :param ori_uval: List containing feature original unique value before feature encoding
    :param encode_uval: List containing feature unique value after feature encoding
    """

    # Display all available menu
    print("""
    MENU:
    1. Add Applicant Data
    2. Remove Applicant
    3. Remove All Applicant
    4. Show Credit Applicant List
    5. Predict Credit Approval
    0. Exit
    """)

    while True:
        # Choose the menu option
        try:
            opt = int(input("Menu [0 - 5]: "))
            if opt < 0 or opt > 5:
                print("Menu {} is not available!".format(opt))
            else:
                break
        except ValueError:
            print("Menu option has to be in integer!")

    # Option 1: Add Applicant Data
    # Adding new applicant details data
    if opt == 1:
        try:
            print("=== ADD NEW APPLICANT DATA ===")

            # Applicant's Key
            name = input("Applicant's Name: ")

            # Feature 1 (Age)
            while True:
                try:
                    age = float(input("Applicant's Age: "))
                except ValueError:
                    print("Age has to be in integer or float!")
                else:
                    break

            # Feature 2 (Debt)
            while True:
                try:
                    debt = float(input("Applicant's Current Debt (USD): "))
                except ValueError:
                    print("Debt (USD) has to be in integer or float!")
                else:
                    break

            # Feature 3 (Married)
            while True:
                print("""
                Marital Status:
                - u = unmarried
                - y = married
                - l = divorced
                """)
                marital_s = input("Applicant's Marital Status [u/y/l]: ")
                if marital_s in ['u', 'y', 'l']:
                    break
                else:
                    print("Inputted marital status is not available!")

            # Feature 4 (BankCustomer)
            while True:
                print("""
                Bank:
                - g = Bank G
                - p = Bank P
                - gg = Bank GG
                """)
                bank = input("Applicant's Bank [g/p/gg]: ")
                if bank in ['g', 'p', 'gg']:
                    break
                else:
                    print("Inputted bank is not available!")

            # Feature 5 (EducationLevel)
            while True:
                print("""
                Education Level:
                - w = Level W
                - q = Level Q
                - m = Level M
                - r = Level R
                - cc = Level CC
                - k = Level K
                - c = Level C
                - d = Level D
                - x = Level X
                - i = Level I
                - e = Level E
                - aa = Level AA
                - ff = Level FF
                - j = Level J
                """)
                education = input("Applicant's Educational Level: ")
                if education in ['w', 'q', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff', 'j']:
                    break
                else:
                    print("Inputted educational level is not available!")

            # Feature 7 (YearsEmployed)
            while True:
                try:
                    years_employed = float(input("Applicant's Years Employed [0 - infinity]: "))
                except ValueError:
                    print("Years employed has to be in integer or float!")
                else:
                    break

            # Feature 8 (PriorDefault)
            while True:
                print("""
                Marital Status:
                - t = previously defaulting on a credit card
                - f = no record of defaulting on a credit card
                """)
                prior_def = input("Prior Default [t/f]: ")
                if prior_def in ['t', 'f']:
                    break
                else:
                    print("Inputted prior default is not available!")

            # Feature 9 (Employed)
            while True:
                print("""
                Marital Status:
                - t = employed
                - f = unemployed
                """)
                employed = input("Employment Status [t/f]: ")
                if employed in ['t', 'f']:
                    break
                else:
                    print("Inputted employment status is not available!")

            # Feature 10 (CreditScore)
            while True:
                try:
                    credit_score = int(input("Credit Score [0 - 67]: "))
                    if credit_score < 0 or credit_score > 67:
                        print("Credit Score {} is not available!".format(credit_score))
                    else:
                        break
                except ValueError:
                    print("Credit score has to be in integer!")

            # Feature 14 (Income)
            while True:
                try:
                    income = float(input("Applicant's Income (USD): "))
                except ValueError:
                    print("Income (USD) has to be in integer or float!")
                else:
                    break

            # Set all the details on a list
            detail = [age, debt, marital_s, bank, education, years_employed, prior_def, employed, credit_score, income]

            # Add applicant by calling function `add_applicant()` on object `applicant`
            applicant.add_applicant(name, detail, ori_uval, encode_uval)

        finally:
            input("Enter any key to continue >>")
            print("\n\n\n")
            menu(applicant, model, ori_uval, encode_uval)

    # Option 2: Remove Applicant
    # Remove specific applicant using applicant's name
    elif opt == 2:
        try:
            applicant.display_applicant()

            if len(applicant.applicant_detail) != 0:
                target = input("Enter applicant's name you want to remove [case sensitive]: ")

                # Remove specific applicant by calling function `delete_applicant()` on object `applicant`
                applicant.delete_applicant(target)
        except KeyError:
            print("""
            FAIL
            Applicant's data is not available!
            """)
        finally:
            input("Enter any key to continue >>")
            print("\n\n\n")
            menu(applicant, model, ori_uval, encode_uval)

    # Option 3: Remove All Applicant
    # Remove all applicant data from the list
    elif opt == 3:
        try:
            applicant.display_applicant()

            while True:
                if len(applicant.applicant_detail) == 0:
                    break

                # Ask for confirmation before removing all applicat data from list
                choice = input("Are you sure you want to remove all applicant data? [Y/N]: ")

                # If 'yes', then remove all applicant by calling function `reset_applicant()` on object `applicant`
                if choice in ['Y', 'y']:
                    applicant.reset_applicant()
                    break
                # Else if no, then the user will be directed back to the program's menu immediately
                elif choice in ['N', 'n']:
                    break
                else:
                    print("Inputted option is not available!")
        finally:
            input("Enter any key to continue >>")
            print("\n\n\n")
            menu(applicant, model, ori_uval, encode_uval)

    # Option 4: Show Credit Applicant List
    # Show details of all available applicant on the list
    elif opt == 4:
        try:
            # Show all available applicant by calling function `display_applicant()` on object `applicant`
            applicant.display_applicant()
        finally:
            input("Enter any key to continue >>")
            print("\n\n\n")
            menu(applicant, model, ori_uval, encode_uval)

    # Option 5: Predict Credit Approval
    # Predicting credit approval from inputted applicant details
    elif opt == 5:
        try:
            # Do prediction by calling function `get_pred_res()` on object `applicant`
            applicant.get_pred_res(model)
        finally:
            input("Enter any key to continue >>")
            print("\n\n\n")
            menu(applicant, model, ori_uval, encode_uval)

    # Option 0: Exit
    # Option to exit the program
    else:
        while True:
            # Ask for confirmation from user before doing the cancellation
            choice = input("Are you sure you are exiting the program? [Y/N]: ")

            # If yes, then exiting the program
            if choice in ['Y', 'y']:
                print("""
                ...exiting program...
                """)
                break
            # Else if no, then the user will be directed back to the program's menu immediately
            elif choice in ['N', 'n']:
                menu(applicant, model, ori_uval, encode_uval)
            else:
                print("Inputted option is not available!")


def main():
    """
    Function that acts as the entry point of the credit approval prediction program
    """

    print("""
    *** WELCOME TO CREDIT CARD APPROVAL SIMULATION ***
    """)

    # Do feature engineering and model training on the dataset named 'Credit Approval'
    print("""
    ...model initialization and training...
    """)
    # Do feature engineering by calling function `feature_engineering()`
    x_train, x_test, y_train, y_test, ori_uval, encode_uval = feature_engineering()

    # Train the model on the train set by calling function `model_training()`
    model = model_training(x_train, y_train)

    # Evaluate the model's performance on random test set and show the performance's result
    acc, _, _, f1 = model_evaluation(model, x_test, y_test)
    print("""
    === MODEL TRAINING DONE ===
    """)
    print("""
    Model Performance Evaluation (on the random test set):
    - Accuracy = {}
    - F1 Score = {}
    """.format(acc, f1))

    # Create new object credit applicant by calling Application() class
    cred_app = Application()

    # Call menu() function to go to the menu
    menu(cred_app, model, ori_uval, encode_uval)

    # Display message saying thank you at the end of the program
    print("""
    *** THANK YOU FOR USING THE PROGRAM ***
    """)


if __name__ == "__main__":
    main()
