import os

import requests
from dotenv import load_dotenv

load_dotenv()

URLHAUS_API_URL = "https://urlhaus-api.abuse.ch/v1/url/"
AUTH_KEY = os.getenv("URLHAUS_AUTH_KEY")


def check_urlhaus(url):
    """
    Check a URL against URLhaus.
    Returns whether URLhaus has a record for the submitted URL.
    """
    if not AUTH_KEY:
        return {
            "available": False,
            "found": False,
            "message": "URLhaus API key is not configured.",
        }

    try:
        response = requests.post(
            URLHAUS_API_URL,
            headers={"Auth-Key": AUTH_KEY},
            data={"url": url},
            timeout=10,
        )

        if response.status_code != 200:
            return {
                "available": False,
                "found": False,
                "message": f"URLhaus returned HTTP {response.status_code}.",
            }

        data = response.json()
        query_status = data.get("query_status")

        if query_status == "ok":
            return {
                "available": True,
                "found": True,
                "message": "URL found in URLhaus.",
                "urlhaus_data": data,
            }

        if query_status == "no_results":
            return {
                "available": True,
                "found": False,
                "message": "URL was not found in URLhaus.",
            }

        # Anything else - some other API response we don't specifically handle
        return {
            "available": True,
            "found": False,
            "message": f"URLhaus response: {query_status}",
            "urlhaus_data": data,
        }

    except requests.RequestException as error:
        return {
            "available": False,
            "found": False,
            "message": f"URLhaus connection failed: {error}",
        }