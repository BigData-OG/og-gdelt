# og-gdelt

Welcome to the GDELT Stock Sentiment Analysis project (`og-gdelt`).

## Architecture

![Architecture Diagram](assets/arch_diagram.png)

### Component Interaction

The architecture consists of several key components working together to ingest, process, and serve sentiment data:

1. **Data Ingestion**: The system continuously ingests news and event data from the GDELT (Global Database of Events, Language, and Tone) project.
2. **Processing & Model Inference**: The ingested data is processed and fed into machine learning models hosted on Google Cloud Vertex AI. These models perform sentiment analysis specifically targeted at stock tickers. The `model_registry` facilitates the management and retrieval of these deployed models.
3. **API Layer**: A backend REST API (located in the `api/` directory) serves as the primary interface for users to query the processed sentiment analysis results. It handles incoming requests, fetches data from the underlying storage or triggers inference, and returns the results.
4. **Infrastructure**: The underlying resources (databases, compute instances, Vertex AI endpoints) are provisioned and managed using the code in the `infrastructure/` directory, ensuring scalable and robust operation on Google Cloud Platform.

---

## API Usage

The API provides endpoints to interact with the system and retrieve sentiment data. Below are examples of how to interact with the API.

*(Note: Update the base URL to your deployed environment's URL)*

### Base URL
`http://localhost:8000/api/v1` (for local development)

### Example Endpoints

#### 1. Infer Stock Price
Predict stock price given company name and ticker

**Request:**
```http
POST /predict
```

**Example via cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json"
```

**Example Response:**
```json
{
  "ticker": "GOOGL",
  "company": "Google"
}
```

#### 2. Train a regression model given company name and ticker
Train a model given company name and ticker

**Request:**
```http
POST /train
```

**Example via cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json"
```

**Example Response:**
```json
{
  "ticker": "GOOGL",
  "company": "Google"
}
```
