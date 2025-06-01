import logging
from typing import List
from PIL import Image, ImageStat
from model.gemini import generate
from logic.models import InfoResponse, ImageMetadata, HistogramData

class ImageInfoAgent:
    """Provides detailed information about images"""
    
    def __init__(self, client=None):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        # Client is not needed since we use the generate functions
        pass
    
    def analyze_image(self, image_path: str, prompt: str) -> InfoResponse:
        """Extract comprehensive image information"""
        
        try:
            # Load image for technical analysis
            img = Image.open(image_path)
            
            # Calculate histogram data
            histogram_data = self._calculate_histogram(img)
            
            # Extract dominant colors
            dominant_colors = self._extract_dominant_colors(img)
            
            # Create metadata object
            metadata = ImageMetadata(
                width=img.width,
                height=img.height,
                format=img.format or "unknown",
                color_space=img.mode,
                channels=len(img.getbands()),
                bit_depth=8  # Assuming standard 8-bit depth, could be calculated more precisely
            )
            
            # Use Gemini for content analysis with optimized prompt for concise response
            analysis_prompt = f"""
            Analyze this image and provide a concise, accurate description in ONE paragraph only.
            
            Focus on: main subjects, scene context, lighting/mood, and key visual elements.
            User question: {prompt}
            
            Keep response under 150 words and directly address the user's specific question while covering essential image details.
            """
            
            self.logger.info("Calling Gemini for image analysis")
            description = generate(
                prompt=analysis_prompt,
                image=image_path,
                system_instruction="You are an expert image analyst. Provide concise, accurate descriptions in exactly one paragraph. Be direct and informative without unnecessary elaboration."
            )
            
            self.logger.info("Image analysis completed")
            
            # Return structured InfoResponse
            return InfoResponse(
                metadata=metadata,
                histogram=histogram_data,
                dominant_colors=dominant_colors,
                description=description.strip()
            )
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
            # Return basic info if analysis fails
            try:
                img = Image.open(image_path)
                metadata = ImageMetadata(
                    width=img.width,
                    height=img.height,
                    format=img.format or "unknown",
                    color_space=img.mode,
                    channels=len(img.getbands()),
                    bit_depth=8
                )
                return InfoResponse(
                    metadata=metadata,
                    histogram=HistogramData(red=[], green=[], blue=[], luminance=[]),
                    dominant_colors=["#000000"],
                    description=f"Image analysis failed: {str(e)}"
                )
            except:
                # Fallback if image can't be loaded
                self.logger.error("Failed to load image for fallback analysis")
                metadata = ImageMetadata(
                    width=0, height=0, format="unknown", 
                    color_space="unknown", channels=0, bit_depth=0
                )
                return InfoResponse(
                    metadata=metadata,
                    histogram=HistogramData(red=[], green=[], blue=[], luminance=[]),
                    dominant_colors=["#000000"],
                    description=f"Failed to load image: {str(e)}"
                )
    
    def _calculate_histogram(self, img: Image.Image) -> HistogramData:
        """Calculate image histogram data"""
        try:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Get histogram data for each channel
            hist = img.histogram()
            r_hist = hist[0:256]
            g_hist = hist[256:512]
            b_hist = hist[512:768]
            
            # Calculate luminance histogram
            lum_img = img.convert('L')
            lum_hist = lum_img.histogram()
            
            # Convert to lists with integers
            return HistogramData(
                red=list(map(int, r_hist)),
                green=list(map(int, g_hist)),
                blue=list(map(int, b_hist)),
                luminance=list(map(int, lum_hist))
            )
        except Exception as e:
            # Return empty histograms if calculation fails
            return HistogramData(red=[], green=[], blue=[], luminance=[])
    
    def _extract_dominant_colors(self, img: Image.Image, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from image"""
        try:
            # Resize for faster processing
            img_small = img.resize((100, 100))
            if img_small.mode != 'RGB':
                img_small = img_small.convert('RGB')
            
            # Simple approach: use average color and create variations
            avg_color = ImageStat.Stat(img_small).mean
            rgb = tuple(map(int, avg_color[:3]))
            
            # Return as hex colors  
            dominant_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            
            # Create some variations of the average for a more diverse palette
            colors = [dominant_color]
            
            # Create darker and lighter variations
            for i in range(1, min(num_colors, 4)):
                if i <= 2:
                    # Darker variations
                    factor = 0.7 - (i * 0.2)
                else:
                    # Lighter variations
                    factor = 1.3 + ((i-2) * 0.2)
                    
                r = max(0, min(255, int(rgb[0] * factor)))
                g = max(0, min(255, int(rgb[1] * factor)))
                b = max(0, min(255, int(rgb[2] * factor)))
                colors.append(f"#{r:02x}{g:02x}{b:02x}")
            
            return colors[:num_colors]
            
        except Exception as e:
            # Return basic black color if extraction fails
            return ["#000000"]