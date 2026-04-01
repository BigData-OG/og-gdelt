from google.cloud import firestore

class ModelRepository:
    def __init__(self, firestore_client: firestore.Client):
        self.firestore_client = firestore_client
        self.collection_name = "gdeltModels"

    def save_model_id(self, company_name, model_id):
        doc_ref = self.firestore_client.collection(self.collection_name).document(company_name)
        doc_ref.set({"modelId": model_id, "companyName": company_name})

    def get_model_id_by_company_name(self, company_name):
        doc_ref = self.firestore_client.collection(self.collection_name).document(company_name)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("modelId")
        return None