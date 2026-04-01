import time
from locust import HttpUser, task, between

class TrainAPILoadTest(HttpUser):
    # Wait time between tasks for a single user
    wait_time = between(1, 5)
    token = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImExMGU1OGRmNTVlNzI4NTY2ZWM1NmJkYTZlYjNiZDQ1NDM5ZjM1ZDciLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTAyMDAyNzM0MTQwMDAzNjMyMzY1IiwiZW1haWwiOiJraW5nc2hhbW1lcnRvbkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXRfaGFzaCI6InJyaWNjRjVJX3AyZWtKcnFDN3lIbFEiLCJpYXQiOjE3NzUwNzAxNjUsImV4cCI6MTc3NTA3Mzc2NX0.JFZ8jlcUmajsxtb-1BhJoyFjvUn3qjketik-fAKNmENo6AD_RFgbz0EnuFJyXpmdxgv3U7RahDTh9uzapNLZzsNLGoiygiFC-gdxovVhXWDNltyHG452OboUstlKx86CLHAz3-el96HubiPHGWZBGaJprto4xW3jRB3VFXkhwj-WuzlWlF4DN1212SQZKLwmPpovURnyrPQJaaVlGg1DslbWEQhU2-FGx7qDMOgFqhWl-VO56bYvHUi6ewiW-P71AeChXyucAFYrKVRfEemnk2DS6tM762Gp702CA8lz5js6z-W1ZVPewpXnA2olhMIkLqWfUorzIQn0Wi-5mAnudA"

    # @task(3)
    # def test_predict_endpoint(self):
    #     """
    #     Test the /predict POST endpoint.
    #     Requires a deployed Vertex AI endpoint to succeed fully,
    #     otherwise it may return 'training_needed'.
    #     """
    #     payload = {
    #         "company_name": "pfizer",
    #         "ticker": "PFE"
    #     }
    #     headers = {
    #         "Authorization": f"Bearer {self.token}",
    #         "Content-Type": "application/json"
    #     }
    #     with self.client.post("/predict", json=payload, headers=headers,catch_response=True) as response:
    #         if response.status_code == 200:
    #             response.success()
    #         else:
    #             response.failure(f"Failed with status code: {response.status_code}. Response: {response.text}")

    @task(3)
    def test_train_endpoint(self):
        """
        Test the /predict POST endpoint.
        Requires a deployed Vertex AI endpoint to succeed fully,
        otherwise it may return 'training_needed'.
        """
        payload = {
            "company_name": "pfizer",
            "ticker": "PFE",
            "skip_extraction": True
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        with self.client.post("/train", json=payload, headers=headers,catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}. Response: {response.text}")
