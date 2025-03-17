from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import json
import requests
from datetime import datetime

# Ollama Model
MODEL_NAME = "gemma3:1b"

# Weather API (No Sign-Up Required)
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# Define function descriptions for LangChain
FUNCTIONS = {
    "get_weather": {
        "description": "Fetches real-time weather data for a given city.",
        "parameters": {
            "city": {"type": "string", "description": "City name"}
        }
    }
}

# Define a structured prompt to force JSON function calls
SYSTEM_PROMPT = """
You are WeatherBot, an AI weather reporter.
When the user asks about weather, respond in this structured JSON format:

{
    "function": "get_weather",
    "parameters": {
        "city": "<city_name>"
    }
}

Examples:
User: What is the weather like in Berlin today?
Assistant: {"function": "get_weather", "parameters": {"city": "Berlin"}}

User: How's the weather in Tokyo?
Assistant: {"function": "get_weather", "parameters": {"city": "Tokyo"}}

For any non-weather questions, respond normally as a helpful assistant.
"""

# Define a prompt for generating weather reports
WEATHER_REPORT_PROMPT = """
You are WeatherBot, a professional weather reporter. Convert the following weather data into a natural, engaging weather report like a meteorologist would deliver. Include the temperature and be conversational.

Weather Data: {weather_data}

City: {city}
"""

def get_weather(city):
    """Fetch detailed weather data from Open-Meteo API."""
    params = {
        "latitude": 51.5074,  # Default London
        "longitude": -0.1278,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "forecast_days": 1
    }

    city_coords = {
        "London": (51.5074, -0.1278),
        "New York": (40.7128, -74.0060),
        "Paris": (48.8566, 2.3522),
        "Tokyo": (35.6762, 139.6503),
        "Sydney": (-33.8688, 151.2093),
        "Berlin": (52.5200, 13.4050),
        "Cairo": (30.0444, 31.2357),
        "Moscow": (55.7558, 37.6173),
        "Beijing": (39.9042, 116.4074),
        "Rio": (-22.9068, -43.1729)
    }

    if city in city_coords:
        params["latitude"], params["longitude"] = city_coords[city]
    else:
        # Default to London if city not found
        city = "Unknown location (defaulting to London)"

    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code == 200:
        return response.json(), city
    else:
        return None, city

def weather_code_to_description(code):
    """Convert WMO weather code to text description."""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown weather condition")

def format_weather_data(weather_data, city):
    """Format raw weather data into a readable string."""
    if not weather_data:
        return f"Unable to retrieve weather data for {city}."
    
    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})
    
    current_temp = current.get("temperature_2m", "N/A")
    feels_like = current.get("apparent_temperature", "N/A")
    humidity = current.get("relative_humidity_2m", "N/A")
    weather_code = current.get("weather_code", 0)
    weather_desc = weather_code_to_description(weather_code)
    wind_speed = current.get("wind_speed_10m", "N/A")
    wind_direction = current.get("wind_direction_10m", "N/A")
    precipitation = current.get("precipitation", 0)
    
    max_temp = daily.get("temperature_2m_max", [0])[0] if "temperature_2m_max" in daily and daily["temperature_2m_max"] else "N/A"
    min_temp = daily.get("temperature_2m_min", [0])[0] if "temperature_2m_min" in daily and daily["temperature_2m_min"] else "N/A"
    
    formatted_data = {
        "city": city,
        "current_temp": current_temp,
        "feels_like": feels_like,
        "conditions": weather_desc,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "precipitation": precipitation,
        "max_temp": max_temp,
        "min_temp": min_temp,
        "time": datetime.now().strftime("%A, %B %d at %I:%M %p")
    }
    
    return formatted_data

def generate_weather_report(weather_data, city, model):
    """Generate a natural language weather report using the model."""
    formatted_data = format_weather_data(weather_data, city)
    
    messages = [
        SystemMessage(content=WEATHER_REPORT_PROMPT.format(
            weather_data=json.dumps(formatted_data, indent=2),
            city=city
        )),
        HumanMessage(content=f"Please give me a weather report for {city}.")
    ]
    
    try:
        response = model.invoke(messages).content
        return response
    except Exception as e:
        # Fallback to basic report if model generation fails
        if isinstance(formatted_data, dict):
            return f"Currently in {formatted_data['city']}, it's {formatted_data['current_temp']}°C with {formatted_data['conditions'].lower()}. The high today will be {formatted_data['max_temp']}°C and the low will be {formatted_data['min_temp']}°C."
        else:
            return formatted_data  # Return the error message

def execute_function(response, model):
    """Extract and execute function calls from model response."""
    try:
        # Extract JSON from code block if present
        if "```json" in response and "```" in response:
            # Find the JSON content between the code block markers
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            json_str = response.strip()
        
        parsed_response = json.loads(json_str)
        function_name = parsed_response["function"]
        parameters = parsed_response.get("parameters", {})

        if function_name == "get_weather":
            city = parameters.get("city", "unknown city")
            weather_data, city_used = get_weather(city)
            return generate_weather_report(weather_data, city_used, model)
        else:
            return f"❌ Unknown function '{function_name}'"
    
    except json.JSONDecodeError:
        # If not a function call, return the original response
        return response

def main():
    """Run the weather reporter AI."""
    model = ChatOllama(model=MODEL_NAME)
    
    print("🌦️ Welcome to AI Weather Reporter! Ask about the weather anywhere.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("AI Weather Reporter: Goodbye! Have a nice day! 👋")
            break
            
        # Send input to LangChain-based Ollama model
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input)
        ]

        print("\nThinking...")
        raw_response = model.invoke(messages).content
        
        # Check if the response looks like a function call
        if "{" in raw_response and "function" in raw_response:
            print("🔍 Fetching weather data...")
            final_response = execute_function(raw_response, model)
        else:
            final_response = raw_response
            
        print(f"\nAI Weather Reporter: {final_response}\n")

if __name__ == "__main__":
    main()