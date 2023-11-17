from training import *

# Used features which has been analysed in `model_training_and_analysis.ipynb`
FEATURE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 14]


def get_idx_ori_encode_dict(dict_ori, feature, target):
    """
    Function to obtain the index of encoded feature value for new applicant's data

    :param dict_ori: List containing feature original unique value before feature encoding
    :param feature: Targeted feature to be used in searching for expected encoded value
    :param target: Targeted value to be used in searching for expected encoded value
    :return: Integer as index of where the targeted value is found in the unique value's dictionary
    """

    if feature in dict_ori and isinstance(dict_ori[feature], list):
        # Get list of all unique value before feature encoding
        val_list = dict_ori[feature]
        if target in val_list:
            # Get the index
            return val_list.index(target)


class Application:
    def __init__(self):
        """
        Method for making applicant list in the class Application.

        :applicant_detail: dictionary which stores the list of applicant details
        """

        self.applicant_detail = dict()

    def add_applicant(self, name, detail_val, ori_uval, encode_uval):
        """
        Function to add new applicant to the applicant list

        :param name: String containing the name of applicant as the key for dictionary
        :param detail_val: List containing the details of applicant
        :param ori_uval: List containing feature original unique value before feature encoding
        :param encode_uval: List containing feature unique value after feature encoding
        :return: Display confirmation in which the applicant details has been added
        """

        # Change some feature value to the encoded feature value to fulfill the pattern of training set
        for i, (key, val) in enumerate(zip(FEATURE, detail_val)):
            # Categorical feature based on analysis in `model_training_and_analysis.ipynb`
            if key in [3, 4, 5, 8, 9]:
                # Get the index of expected encoded feature value by calling function `get_idx_ori_encode_dict()`
                idx = get_idx_ori_encode_dict(ori_uval, key, val)
                # Get the encoded feature value
                val = encode_uval[key][idx]
                # Change the feature value to the encoded feature value
                detail_val[i] = val

        # Add the new applicant data to the applicant list
        applicant_dict = {key: val for key, val in zip(FEATURE, detail_val)}
        self.applicant_detail.update({name: applicant_dict})

        # Display confirmation that the applicant details has been succesfully added
        return print("""
        SUCCESS
        Applicant {} is already added!
        """.format(name))

    def get_applicant_df(self):
        """
        Function to get the DataFrame from applicant list

        :return: DataFrame if the applicant_detail is not empty, else return confirmation that the list is empty
        """

        if len(self.applicant_detail) == 0:
            return "=== CREDIT APPLICANT LIST IS EMPTY ==="
        else:
            show_df = pd.DataFrame.from_dict(self.applicant_detail, orient='index')

            return show_df

    def display_applicant(self):
        """
        Function to display the DataFrame of applicant details

        :return: Display the DataFrame of applicant details
        """

        print("=== CREDIT APPLICANT LIST ===")
        print(self.get_applicant_df())

    def delete_applicant(self, name):
        """
        Function to remove specific applicant based on the given name

        :param name: String containing the name that want to be removed from applicant list
        :return: Display confirmation in which the targeted applicant has been removed
        """

        # Remove applicant detail which has `name` as the key
        self.applicant_detail.pop(name)

        return print("""
        SUCCESS
        Applicant {} is already removed!
        """.format(name))

    def reset_applicant(self):
        """
        Function to remove all applicant from applicant list

        :return: Display confrimation in which all applicants have been removed
        """

        # Remove all applicants from the list
        self.applicant_detail.clear()

        return print("""
        SUCCESS
        All applicant data is already removed!
        """)

    def show_pred_res(self, pred_res):
        """
        Function to show the prediction result of applicant's credit approval

        :param pred_res: Array containing the result of applicant's credit approval prediction
        :return: Display the prediction result (APPROVED or DISAPPROVED) of applicant's credit approval
        """

        applicant_name = list(self.applicant_detail.keys())

        # If `pred_res` = 1, then display 'APPROVED', else display 'DISAPPROVED'
        for i, pred_res in enumerate(pred_res):
            if pred_res == 1:
                res = 'APPROVED'
            else:
                res = 'DISAPPROVED'
            print("{}. Approval Prediction for Applicant '{}' = {}".format((i + 1), applicant_name[i], res))

    def get_pred_res(self, model):
        """
        Function to do prediction using the trained model on new applicant's details

        :param model: Random Forest model which has been trained before
        :return: Display the prediction result if the applicant list is not empty, else return that the list is empty
        """

        if len(self.applicant_detail) == 0:
            print("=== CREDIT APPLICANT LIST IS EMPTY ===")
        else:
            print("=== CREDIT APPROVAL PREDICTION ===")
            # Get the DataFrame containing the applicant's detail
            x_pred = self.get_applicant_df()

            # Predict the applicant's credit approval using the trained model
            pred_res = model.predict(x_pred)

            # Show the prediction result by calling function `show_pred_res()`
            self.show_pred_res(pred_res)
