import requests
from bs4 import BeautifulSoup

class Esports8Scraper:
    def __init__(self):
        self.base_url = "https://www.esports8.com/en/lol/"

    def fetch_actual_results(self):
        """Scrapes player results from Esports8 and returns a dictionary of player stats."""
        response = requests.get(self.base_url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            print("Failed to fetch Esports8 data")
            return {}

        soup = BeautifulSoup(response.text, "html.parser")
        player_results = {}

        # Modify these selectors based on the site's HTML structure
        players = soup.select(".player-stat")  # Example selector, update based on site

        for player in players:
            name = player.select_one(".player-name").text.strip()
            actual_stat = player.select_one(".stat-value").text.strip()  # Adjust selector
            player_results[name] = actual_stat

        return player_results

# Run scraper independently if needed
if __name__ == "__main__":
    scraper = Esports8Scraper()
    results = scraper.fetch_actual_results()
    print(results)  # Print results for debugging
