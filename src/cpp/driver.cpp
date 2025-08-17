#include <iostream>
#include <cstdlib>   // For system()
#include <fstream>   // For ifstream

using namespace std;

bool fileExists(const string& filename) {
    ifstream f(filename);
    return f.good();
}

int main() {
    cout << "=== Loan Prediction System ===\n";

    string base_path = "E:/Loan_Predictor/Loan-Prediction-System"; // change this

    string model_file   = base_path + "/src/python/saved_models/loan_model.pkl";
    string train_script = "python " + base_path + "/src/python/train_model.py";
    string predict_script = "python " + base_path + "/src/python/predict.py";

    // Step 1: Train if model file not found
    if (!fileExists(model_file)) {
        cout << "\nModel not found. Training model...\n";
        if (system(train_script.c_str()) != 0) {
            cerr << "Error: Failed to train model.\n";
            return 1;
        }
    } else {
        cout << "\nModel already exists. Skipping training.\n";
    }

    // Step 2: Predict
    cout << "\nStarting prediction...\n";
    if (system(predict_script.c_str()) != 0) {
        cerr << "Error: Prediction script failed.\n";
        return 1;
    }

    cout << "\n=== Done ===\n";
    return 0;
}
