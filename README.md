# PerPain-allocation
This is the R shiny app used to allocate patients based on their characteristics to three different treatments during the PerPain project.

To run the code, you will need to enter your URI and token from the REDCAP system as well as modify the postForm function.
The code will attempt to install all required packages on its first run.

The app uses a pretrained XGBoost model to make predictions about the treatment assignment as well as visually display the patient's profile.
