# tools/google_trends_tool.py

from crewai_tools.tools.base_tool import BaseTool
from pytrends.request import TrendReq
from requests.exceptions import Timeout

class AdvancedGoogleTrendsTool(BaseTool):
    name: str = "Advanced Google Trends Analyzer"
    description: str = """
    Comprehensive Google Trends analysis tool that provides:
    - Interest over time with detailed timeframes
    - Related topics and queries
    - Regional interest breakdown
    - Rising and top searches
    - Comparative analysis with related terms
    """

    def _run(self, query: str):
        pytrends = TrendReq()
        try:
            pytrends.build_payload([query])
            return {
                "interest_over_time": pytrends.interest_over_time().reset_index().to_dict(orient="records"),
                "related_queries": pytrends.related_queries().get(query, {}),
                "regional_interest": pytrends.interest_by_region().reset_index().to_dict(orient="records")
            }
        except Timeout as e:
            return {"error": f"Request to Google Trends timed out. {e}"}
        except Exception as e:
            return {"error": f"Some other error occured {e}"}


i = AdvancedGoogleTrendsTool()
print(i._run("iphone 17 price"))