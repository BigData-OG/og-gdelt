from google.cloud import aiplatform

PROJECT_ID = "gdelt-stock-sentiment-analysis"
REGION = "us-west1"


def get_models_on_vertex_ai(project: str = PROJECT_ID, region: str = REGION):
    """Gets a dictionary of all models on Vertex AI.

    Args:
        project: The Google Cloud project ID.
        region: The Google Cloud region.

    Returns:
        A dictionary mapping model IDs to model display names.
    """
    # Init Vertex AI - by default, same region and project as above
    aiplatform.init(project=project, location=region)

    models = aiplatform.Model.list()

    model_dict = {}  # map model_id -> display_name

    for model in models:
        # model.resource_name looks like: projects/.../models/123456
        model_id = model.resource_name.split("/")[-1]
        model_dict[model_id] = model.display_name

    return model_dict
