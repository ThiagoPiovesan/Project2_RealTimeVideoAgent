import os
import io
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

class VideoAnalyticAgent:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.model = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=self.api_key)

    def analyze_image(self, image_data: bytes, box: tuple[float, float, float, float], track_id: int) -> str:
        """
        Analyzes a single image and returns a string description of its elements.

        Args:
            image_data: The image data as bytes.
            box: A tuple representing the bounding box coordinates (x, y, width, height).
            track_id: An integer representing the track ID.
        Returns:
            A string describing the elements present in the image.
        """
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Describe the element present in this image in detail. The bounding box coordinates are {box}."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
            ]
        )
        response = self.model.invoke([message])
        return f"For track ID {track_id}: {response.content}"

    def analyze_video_frame(self, frame_data: bytes, box: tuple[float, float, float, float], track_id: int) -> str:
        """
        Analyzes a single video frame (treated as an image) and returns a string description.

        Args:
            frame_data: The video frame data as bytes.
            box: A tuple representing the bounding box coordinates (x, y, width, height).
            track_id: An integer representing the track ID.
        Returns:
            A string describing the elements present in the video frame.
        """
        return self.analyze_image(frame_data, box, track_id)

    # Note: For analyzing a "short video" (multiple frames), you would typically
    # process each frame individually or send a sequence of frames if the API
    # supports it. The current Gemini Vision Pro model is primarily designed
    # for image