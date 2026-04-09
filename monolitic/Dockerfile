# Use an official, lightweight Python runtime
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY main.py .
COPY train_routes.py .
COPY train.py .
COPY train_results_vertex_model_amazon_model_model.joblib .
COPY joined_data_combined_data_clean.csv .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
