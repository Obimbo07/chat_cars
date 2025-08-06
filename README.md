Cars Dataset Retrieval Chatbot
A retrieval-based chatbot that queries a cars dataset using LangChain and FAISS.
Overview
This project implements a retrieval-only chatbot that allows users to query a dataset of cars containing information on car companies, models, engine types, CC/battery capacity, horsepower, top speed, 0-100 km/h performance, price, fuel type, seating capacity, and torque. It uses LangChain for document processing and FAISS as the vector store with HuggingFace embeddings for efficient retrieval. The chatbot accepts natural language questions via a command-line interface and returns relevant car entries.
Installation

Clone the repository:
git clone <repository-url>
cd cars-chatbot


Install dependencies:
pip install -r requirements.txt


Place the cars dataset CSV file (cars_dataset.csv) in the data/ directory.


Usage
Run the chatbot with:
python rag_cars_chatbot.py --csv_path data/cars_dataset.csv

Example queries:

"Which cars have over 300 horsepower?"
"Show me electric cars with a top speed above 200 km/h."
"What are the cheapest cars with seating for 7?"

Type exit to quit the chatbot.
License
MIT License. See LICENSE for details.
Data Requirements

The dataset must be a CSV file named cars_dataset.csv with columns: Car_Company, Car_Model, Engine_Type, CC_Battery_Capacity, Horsepower_HP, Top_Speed, Zero_To_Hundred, Price_USD, Fuel_Type, Seating_Capacity, Torque.
Place the CSV file in the data/ directory.
The dataset is for educational and research purposes only and should not be used for commercial purposes without authorization.
