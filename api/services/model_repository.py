import os
from google.cloud import firestore

class ModelRepository:
    _instance = None

    def __new__(cls, firestore_client: firestore.Client = None):
        if cls._instance is None:
            cls._instance = super(ModelRepository, cls).__new__(cls)
            # Initialize strictly once
            cls._instance.firestore_client = firestore_client
            cls._instance.collection_name = os.environ.get("FIRESTORE_COLLECTION_NAME", "gdeltModels")
        return cls._instance

    def save_model_id(self, company_name, model_id):
        doc_ref = self.firestore_client.collection(self.collection_name).document(company_name)
        doc_ref.set({"modelId": model_id, "companyName": company_name})

    def get_model_id_by_company_name(self, company_name):
        doc_ref = self.firestore_client.collection(self.collection_name).document(company_name)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("modelId")
        return None