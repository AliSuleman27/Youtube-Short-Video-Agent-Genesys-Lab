from crewai_tools import BaseTool
from typing import Type, Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
import pytrends
from pytrends.request import TrendReq
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import logging

class GoogleTrendsInput(BaseModel):
    """Input schema for Google Trends analysis."""
    keywords: Union[str, List[str]] = Field(
        ..., 
        description="Single keyword or list of keywords to analyze (max 5 for comparison)"
    )
    timeframe: str = Field(
        default="today 12-m", 
        description="Timeframe for analysis (e.g., 'today 12-m', 'today 5-y', '2023-01-01 2023-12-31')"
    )
    geo: str = Field(
        default="", 
        description="Geographic location (e.g., 'US', 'GB', 'DE', '' for worldwide)"
    )
    category: int = Field(
        default=0, 
        description="Category ID (0 for all categories, see Google Trends categories)"
    )
    gprop: str = Field(
        default="", 
        description="Google property ('', 'images', 'news', 'youtube', 'froogle')"
    )
    include_related: bool = Field(
        default=True, 
        description="Include related topics and queries in analysis"
    )
    include_regional: bool = Field(
        default=True, 
        description="Include regional interest breakdown"
    )
    include_rising: bool = Field(
        default=True, 
        description="Include rising searches analysis"
    )

class AdvancedGoogleTrendsAnalyzer(BaseTool):
    name: str = "Advanced Google Trends Analyzer"
    description: str = """
    Comprehensive Google Trends analysis tool that provides:
    - Interest over time with detailed timeframes
    - Related topics and queries
    - Regional interest breakdown
    - Rising and top searches
    - Comparative analysis with related terms
    """
    args_schema: Type[BaseModel] = GoogleTrendsInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, 
             keywords: Union[str, List[str]],
             timeframe: str = "today 12-m",
             geo: str = "",
             category: int = 0,
             gprop: str = "",
             include_related: bool = True,
             include_regional: bool = True,
             include_rising: bool = True) -> str:
        """
        Execute comprehensive Google Trends analysis.
        
        Args:
            keywords: Single keyword or list of keywords to analyze
            timeframe: Time period for analysis
            geo: Geographic location code
            category: Category ID for filtering
            gprop: Google property to search
            include_related: Whether to include related topics/queries
            include_regional: Whether to include regional breakdown
            include_rising: Whether to include rising searches
            
        Returns:
            JSON string with comprehensive trends analysis
        """
        try:
            # Initialize pytrends and logger within the method
            pytrends = TrendReq(hl='en-US', tz=360, retries=3, backoff_factor=0.5)
            logger = logging.getLogger(__name__)
            
            # Normalize keywords to list
            if isinstance(keywords, str):
                kw_list = [keywords]
            else:
                kw_list = keywords[:5]  # Limit to 5 keywords for API constraints
            
            # Build payload for pytrends
            pytrends.build_payload(
                kw_list=kw_list,
                cat=category,
                timeframe=timeframe,
                geo=geo,
                gprop=gprop
            )
            
            analysis_result = {
                "query_info": {
                    "keywords": kw_list,
                    "timeframe": timeframe,
                    "geo": geo if geo else "Worldwide",
                    "category": category,
                    "google_property": gprop if gprop else "Web Search",
                    "analysis_date": datetime.now().isoformat()
                },
                "interest_over_time": {},
                "related_topics": {},
                "related_queries": {},
                "regional_interest": {},
                "rising_searches": {},
                "summary": {}
            }
            
            # Get interest over time
            try:
                interest_df = pytrends.interest_over_time()
                if not interest_df.empty:
                    # Remove 'isPartial' column if it exists
                    if 'isPartial' in interest_df.columns:
                        interest_df = interest_df.drop('isPartial', axis=1)
                    
                    analysis_result["interest_over_time"] = {
                        "data": interest_df.to_dict('records'),
                        "peak_periods": self._find_peak_periods(interest_df),
                        "trend_direction": self._analyze_trend_direction(interest_df),
                        "average_interest": interest_df.mean().to_dict()
                    }
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Could not retrieve interest over time: {str(e)}")
            
            # Get related topics and queries for each keyword
            if include_related:
                for keyword in kw_list:
                    try:
                        # Related topics
                        related_topics = pytrends.related_topics()
                        if keyword in related_topics and related_topics[keyword] is not None:
                            analysis_result["related_topics"][keyword] = {
                                "top": self._process_related_data(related_topics[keyword].get('top')),
                                "rising": self._process_related_data(related_topics[keyword].get('rising'))
                            }
                        
                        time.sleep(1)  # Rate limiting
                        
                        # Related queries
                        related_queries = pytrends.related_queries()
                        if keyword in related_queries and related_queries[keyword] is not None:
                            analysis_result["related_queries"][keyword] = {
                                "top": self._process_related_data(related_queries[keyword].get('top')),
                                "rising": self._process_related_data(related_queries[keyword].get('rising'))
                            }
                        
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"Could not retrieve related data for {keyword}: {str(e)}")
            
            # Get regional interest
            if include_regional:
                try:
                    regional_df = pytrends.interest_by_region(resolution='COUNTRY')
                    if not regional_df.empty:
                        analysis_result["regional_interest"] = {
                            "by_country": regional_df.to_dict('index'),
                            "top_regions": self._get_top_regions(regional_df)
                        }
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Could not retrieve regional interest: {str(e)}")
            
            # Get rising searches if requested
            if include_rising:
                try:
                    # This uses trending searches which might not be available for all regions
                    trending_searches = pytrends.trending_searches(pn='united_states')
                    if not trending_searches.empty:
                        analysis_result["rising_searches"]["trending_now"] = trending_searches[0].head(10).tolist()
                except Exception as e:
                    logger.warning(f"Could not retrieve rising searches: {str(e)}")
            
            # Generate summary insights
            analysis_result["summary"] = self._generate_summary(analysis_result, kw_list)
            
            return json.dumps(analysis_result, indent=2, default=str)
            
        except Exception as e:
            error_result = {
                "error": f"Analysis failed: {str(e)}",
                "keywords": kw_list if 'kw_list' in locals() else keywords,
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(error_result, indent=2)
    
    def _process_related_data(self, data) -> List[Dict]:
        """Process related topics/queries data."""
        if data is None or data.empty:
            return []
        
        # Convert to list of dictionaries
        result = []
        for _, row in data.head(10).iterrows():  # Limit to top 10
            if 'topic_title' in row:  # Related topics
                result.append({
                    "title": row.get('topic_title', ''),
                    "type": row.get('topic_type', ''),
                    "value": row.get('value', 0)
                })
            else:  # Related queries
                result.append({
                    "query": row.get('query', ''),
                    "value": row.get('value', 0)
                })
        return result
    
    def _find_peak_periods(self, df) -> Dict:
        """Identify peak periods in the data."""
        peaks = {}
        for column in df.columns:
            if column != 'isPartial':
                max_idx = df[column].idxmax()
                max_value = df[column].max()
                peaks[column] = {
                    "peak_date": max_idx.strftime('%Y-%m-%d') if hasattr(max_idx, 'strftime') else str(max_idx),
                    "peak_value": float(max_value)
                }
        return peaks
    
    def _analyze_trend_direction(self, df) -> Dict:
        """Analyze overall trend direction."""
        trends = {}
        for column in df.columns:
            if column != 'isPartial':
                # Calculate trend using first and last 10% of data
                data_length = len(df)
                start_avg = df[column].head(max(1, data_length // 10)).mean()
                end_avg = df[column].tail(max(1, data_length // 10)).mean()
                
                if end_avg > start_avg * 1.1:
                    direction = "Rising"
                elif end_avg < start_avg * 0.9:
                    direction = "Declining"
                else:
                    direction = "Stable"
                
                trends[column] = {
                    "direction": direction,
                    "start_average": float(start_avg),
                    "end_average": float(end_avg),
                    "change_percentage": float((end_avg - start_avg) / start_avg * 100) if start_avg > 0 else 0
                }
        return trends
    
    def _get_top_regions(self, df) -> Dict:
        """Get top regions for each keyword."""
        top_regions = {}
        for column in df.columns:
            # Sort by values and get top 10
            top_10 = df[column].sort_values(ascending=False).head(10)
            top_regions[column] = [
                {"region": region, "interest": float(value)} 
                for region, value in top_10.items() if value > 0
            ]
        return top_regions
    
    def _generate_summary(self, analysis_result: Dict, keywords: List[str]) -> Dict:
        """Generate summary insights from the analysis."""
        summary = {
            "key_insights": [],
            "recommendations": [],
            "data_quality": {}
        }
        
        # Analyze interest over time
        if analysis_result["interest_over_time"]:
            interest_data = analysis_result["interest_over_time"]
            
            for keyword in keywords:
                if keyword in interest_data.get("average_interest", {}):
                    avg_interest = interest_data["average_interest"][keyword]
                    trend_info = interest_data.get("trend_direction", {}).get(keyword, {})
                    
                    summary["key_insights"].append(
                        f"{keyword}: Average interest of {avg_interest:.1f}, "
                        f"trend is {trend_info.get('direction', 'unknown').lower()}"
                    )
        
        # Analyze regional data
        if analysis_result["regional_interest"]:
            summary["key_insights"].append(
                f"Regional analysis available for {len(analysis_result['regional_interest'].get('by_country', {}))} countries"
            )
        
        # Data quality assessment
        summary["data_quality"] = {
            "has_time_series": bool(analysis_result["interest_over_time"]),
            "has_related_data": bool(analysis_result["related_topics"] or analysis_result["related_queries"]),
            "has_regional_data": bool(analysis_result["regional_interest"]),
            "keywords_analyzed": len(keywords)
        }
        
        # Recommendations
        if len(keywords) == 1:
            summary["recommendations"].append("Consider comparing with related keywords for better insights")
        
        if not analysis_result["regional_interest"]:
            summary["recommendations"].append("Try regional analysis to understand geographic patterns")
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Initialize the tool
    trends_tool = AdvancedGoogleTrendsAnalyzer()
    
    # Example analysis
    result = trends_tool._run(
        keywords=["artificial intelligence", "machine learning"],
        timeframe="today 12-m",
        geo="US",
        include_related=True,
        include_regional=True
    )
    
    print("Google Trends Analysis Result:")
    print(result)