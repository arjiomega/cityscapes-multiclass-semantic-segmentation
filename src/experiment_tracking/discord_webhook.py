from datetime import datetime
from zoneinfo import ZoneInfo
import requests


class DiscordWebhook:
    def __init__(self, webhook_url, username, avatar_url):
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url

    def send_message(self, content, embeds=None, files=None):
        data = {
            "content": content,
            "username": self.username,
            "avatar_url": self.avatar_url,
            "embeds": embeds if embeds else []
        }

        response = requests.post(self.webhook_url, json=data, files=files)

        # Handle different status codes
        if files and response.status_code != 200:
            raise ValueError(f"Failed to send message with file: {response.status_code}, {response.text}")
        elif not files and response.status_code != 204:
            raise ValueError(f"Failed to send message: {response.status_code}, {response.text}")
        
        print("Message sent successfully")

    def send_embed(self, title, description, color=0x00ff00, fields=None, footer_text=None, timestamp=None):
        """
        Send an embed message to the Discord channel.

        Args:
            title (str): The title of the embed.
            description (str): The description of the embed.
            color (int): The color of the embed.
            fields (list): A list of field objects.
            footer_text (str): The footer text.
            timestamp (str): The timestamp in ISO format.
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields if fields else [],
            "footer": {"text": footer_text} if footer_text else {},
            "timestamp": timestamp
        }

        self.send_message(content=None, embeds=[embed])

if __name__ == "__main__":
    webhook_url = input("Enter the Discord Webhook URL: ")
    discord_webhook = DiscordWebhook(webhook_url, username="Training Bot", avatar_url="https://letsenhance.io/static/8f5e523ee6b2479e26ecc91b9c25261e/1015f/MainAfter.jpg")
    
    # Send a simple message
    discord_webhook.send_message("Hello, this is a test message from the bot!")

    # Send an embed message
    discord_webhook.send_embed(
        title="Training Complete",
        description="The training process has completed successfully.",
        fields=[
            {"name": "Epochs", "value": "10", "inline": True},
            {"name": "Final Loss", "value": "0.1234", "inline": True}
        ],
        footer_text="Training Bot",
        timestamp="2023-10-01T12:34:56.789Z"
    )