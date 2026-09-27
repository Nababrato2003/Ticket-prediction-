# AI Ticket Classification System

## 1. Project Overview
This project is an automated AI/ML-based system designed to classify incoming customer support tickets. It reads customer complaints, performs natural language text preprocessing, trains a machine learning model, and predicts the correct ticket category. The project includes modular source code for training and prediction, as well as an interactive web dashboard.

## 2. Problem Statement
Customer support teams receive thousands of requests across various channels (email, chat, etc.). Manually reading and sorting these tickets into categories (such as *Hardware issue*, *Software bug*, or *Refund request*) is time-consuming and inefficient. The goal of this project is to automatically predict the category of a new support ticket based strictly on its description.

## 3. Technologies Used
- **Python 3**: Core programming language.
- **Streamlit**: For creating the interactive web application UI.
- **Scikit-Learn**: For machine learning algorithms (Logistic Regression, TF-IDF Vectorization, Evaluation Metrics).
- **Pandas & NumPy**: For data manipulation and analysis.
- **Plotly, Matplotlib, Seaborn**: For exploratory data analysis (EDA) and data visualization.

## 4. Dataset Information & Exploratory Data Analysis (EDA)
The dataset used is `tickets.csv`, which contains historical customer support data. 
- **Input Feature:** `Ticket Description` (the text body of the complaint).
- **Target Variable:** `Ticket Category` (the department or issue type).
- **Other Columns:** Includes metadata like Ticket Priority, Ticket Status, Customer Name, and Product Purchased.

**Key EDA Observations:**
- **Category Chart:** Visualizing the ticket categories reveals the most common problems our customers face (e.g., whether we get more hardware complaints vs. software bugs), allowing the business to allocate support agents where they are needed most.
- **Priority Chart:** The priority distribution (Critical, High, Low) highlights the overall urgency of incoming tickets, ensuring the company can manage its resources to handle critical issues quickly.

## 5. Installation Steps
1. Clone or download this project to your local machine.
2. Open a terminal and navigate to the root directory of the project (`AI_Ticket_Classification`).
3. Ensure you have Python installed.

## 6. Required Dependencies
Install all required libraries using the provided `requirements.txt` file. Run the following command:
```bash
pip install -r requirements.txt
```

## 7. How to Train the Model
To manually train the machine learning model and generate the `.pkl` files, run the training script from the `src/` directory:
```bash
cd src
python train.py
```
*This script will process the dataset, train the TF-IDF vectorizer and Logistic Regression model, and save them into the `model/` directory.*

## 8. How to Perform Prediction
To test the model's predictions via the command line on predefined test cases, run the prediction script from the `src/` directory:
```bash
cd src
python predict.py
```

## 9. How to Run the Application
To launch the interactive web interface, return to the root folder of the project and run the Streamlit app:
```bash
python -m streamlit run app.py
```
*The dashboard will automatically open in your web browser at `http://localhost:8501`, allowing you to analyze data and type in new ticket descriptions.*

## 10. Sample Input/Output

**Sample 1:**
- **Input:** *"My device's battery drains incredibly fast after the recent software update."*
- **Output:** Predicted Category: `Battery life` (Confidence: 97.4%)

**Sample 2:**
- **Input:** *"I am unable to login, password reset is not working"*
- **Output:** Predicted Category: `Account access` (Confidence: 89.2%)

**Sample 3:**
- **Input:** *"I want a refund, the product is damaged"*
- **Output:** Predicted Category: `Refund request` (Confidence: 92.5%)

## 11. Model Evaluation & Results
Understanding the machine learning metrics in simple language:
- **Accuracy (~97%):** Out of 100 tickets, the model guesses the exact correct category 97 times.
- **Precision (~97%):** When the model says a ticket is a "Hardware issue", it is right 97% of the time (it rarely cries wolf).
- **Recall (~97%):** Out of all the actual "Hardware issue" tickets that exist, the model successfully finds and catches 97% of them (it rarely misses any).
- **F1 Score (~97%):** This is just an overall score that combines Precision and Recall to give us a single grade of how balanced the model is.
- **Confusion Matrix:** This is a grid picture that shows exactly where the model gets confused. A perfect model has a dark diagonal line down the middle (which ours does!).

**Is the model performing satisfactorily?**
**Yes**, from a purely mathematical standpoint, the model is performing exceptionally well. Achieving ~97% across all metrics indicates that the algorithm has successfully adapted to the high-dimensional text data and memorized the structural patterns of the provided dataset. However, because the dataset itself is synthetic, the model's logic is constrained to this specific dataset's vocabulary, which represents a limitation for real-world deployment.

## 12. Generative AI Bonus: Automated Rule-Based Responses
This project includes a bonus Generative AI feature inside the Streamlit Web Application. Based on the Machine Learning prediction, the system utilizes a rule-based engine to instantly generate a predefined, helpful response to the customer.

**Example Scenario:**
- **Customer Types:** *"My device's battery drains incredibly fast after the recent software update."*
- **Model Predicts:** `Battery life`
- **System Automatically Suggests:** *"Battery degradation can happen over time. We recommend turning off background app refresh. If the device is under warranty, we can schedule a battery replacement."*
